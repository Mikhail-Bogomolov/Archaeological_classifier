from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse, Response
from fastapi import UploadFile, File, Form, Query
from datetime import datetime
import io
import csv

from PIL import Image

from app import db
from app.inference import run_inference


app = FastAPI()

templates = Jinja2Templates(directory="app/templates")
app.mount("/static", StaticFiles(directory="app/static"), name="static")

pending_scan = None


@app.get("/")
async def home(request: Request, page: int = Query(1, ge=1)):
    db.init_db()
    meta = db.pagination_meta(page)
    objects = db.list_objects_paginated(meta["page"], meta["per_page"])
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "objects": objects,
            "total_count": meta["total_count"],
            "page": meta["page"],
            "total_pages": meta["total_pages"],
            "has_prev": meta["has_prev"],
            "has_next": meta["has_next"],
            "prev_page": meta["prev_page"],
            "next_page": meta["next_page"],
            "per_page": meta["per_page"],
        }
    )


@app.get("/scan")
async def scan_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="scan.html",
        context={"scan": pending_scan}
    )


@app.post("/scan")
async def perform_scan(
    file: UploadFile = File(...),
    object_name: str = Form(None),
):
    global pending_scan
    contents = await file.read()
    pending_scan = {
        **run_inference(contents, object_name),
        "image_bytes": contents,
        "image_mime": file.content_type or "application/octet-stream",
        "image_url": "/scan/image",
    }
    return RedirectResponse("/scan", status_code=303)


@app.post("/scan/rescan")
async def rescan():
    global pending_scan
    pending_scan = None
    return RedirectResponse("/scan", status_code=303)


@app.post("/scan/add")
async def add_scanned_object():
    global pending_scan
    if pending_scan is None:
        return RedirectResponse("/scan", status_code=303)

    db.init_db()
    db.add_object(
        name=str(pending_scan["name"]),
        description=str(pending_scan["description"]),
        category=str(pending_scan["category"]),
        confidence=int(pending_scan["confidence"]),
        date=datetime.now().strftime("%d.%m.%Y %H:%M"),
        features=list(pending_scan.get("features") or []),
        image_bytes=bytes(pending_scan["image_bytes"]),
        image_mime=str(pending_scan.get("image_mime") or "application/octet-stream"),
    )
    pending_scan = None
    return RedirectResponse("/", status_code=303)


@app.get("/object/{object_id}")
async def object_detail(request: Request, object_id: int):
    db.init_db()
    obj = db.get_object(object_id)
    if obj is None:
        return RedirectResponse("/", status_code=303)
    obj["image_url"] = f"/object/{object_id}/image"
    return templates.TemplateResponse(
        request=request,
        name="object_detail.html",
        context={"obj": obj},
    )


@app.get("/scan/image")
async def pending_scan_image():
    if pending_scan is None:
        return RedirectResponse("/scan", status_code=303)
    return StreamingResponse(
        io.BytesIO(pending_scan["image_bytes"]),
        media_type=str(pending_scan.get("image_mime") or "application/octet-stream"),
    )


@app.get("/object/{object_id}/image")
async def object_image(object_id: int):
    db.init_db()
    data = db.get_object_image(object_id)
    if data is None:
        return RedirectResponse("/", status_code=303)
    image_bytes, image_mime = data
    return StreamingResponse(io.BytesIO(image_bytes), media_type=image_mime)


@app.get("/export/csv")
async def export_csv():
    db.init_db()
    objects = db.list_objects()
    out = io.StringIO()
    writer = csv.DictWriter(
        out,
        fieldnames=["id", "name", "description", "category", "confidence", "date"],
        extrasaction="ignore",
    )
    writer.writeheader()
    for obj in objects:
        writer.writerow(obj)
    data = out.getvalue().encode("utf-8-sig")
    return Response(
        content=data,
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=objects.csv"},
    )


@app.get("/object/{object_id}/thumbnail")
async def object_thumbnail(object_id: int, size: int = Query(48, ge=32, le=200)):
    db.init_db()
    data = db.get_object_image(object_id)
    if data is None:
        return RedirectResponse("/", status_code=303)

    image_bytes, _image_mime = data
    img = Image.open(io.BytesIO(bytes(image_bytes)))

    if img.mode in ("RGBA", "P", "LA"):
        if img.mode == "P":
            img = img.convert("RGBA")
        if img.mode in ("RGBA", "LA"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "LA":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1])
            img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    target_w, target_h = size, max(1, int(size * 0.75))
    img_w, img_h = img.size
    aspect_img = img_w / img_h
    aspect_target = target_w / target_h

    if aspect_img > aspect_target:
        new_w = int(img_h * aspect_target)
        left = (img_w - new_w) // 2
        img = img.crop((left, 0, left + new_w, img_h))
    else:
        new_h = int(img_w / aspect_target)
        top = (img_h - new_h) // 2
        img = img.crop((0, top, img_w, top + new_h))

    img = img.resize((target_w, target_h), Image.LANCZOS)
    output = io.BytesIO()
    img.save(output, format="WEBP", quality=75, method=6)
    output.seek(0)
    return StreamingResponse(output, media_type="image/webp")
