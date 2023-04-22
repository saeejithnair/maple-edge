from typing import Optional
import asyncio
import os
from typing_extensions import Annotated

from fastapi import FastAPI, HTTPException, Response, Query, Path
from fastapi.responses import FileResponse, JSONResponse
from sqlmodel import select
from sqlalchemy.exc import MultipleResultsFound

from maple.db.models import NASBench201Model, NASBench201ModelBase
from maple.db import database as db
from sqlmodel.ext.asyncio.session import AsyncSession

database = "maple_test_db"
database_url = db.get_database_url(database=database)

async_engine = db.create_engine(database_url)

app = FastAPI()

@app.on_event("startup")
async def on_startup():
    await db.create_db_and_tables(engine=async_engine)

@app.post("/models/", response_model=NASBench201Model)
async def create_model(model: NASBench201Model):
    async with db.get_session(async_engine) as session:
        session.add(model)
        await session.commit()
        await session.refresh(model)
        return model

async def query_model_by_id(session: AsyncSession,
                            model_id: int) -> NASBench201Model:
    model = await session.get(NASBench201Model, model_id)

    if not model:
        raise HTTPException(status_code=404,
                            detail=f"Model ID: {model_id} not found")

    return model

async def query_model_by_index(session: AsyncSession, dataset_version: float,
                               framework: str, input_dim: int, name: str,
                               type: str, device: str) -> NASBench201Model:
    statement = select(NASBench201Model).where(
        NASBench201Model.name == name,
        NASBench201Model.framework == framework,
        NASBench201Model.input_dim == input_dim,
        NASBench201Model.device == device,
        NASBench201Model.model_type == type,
        NASBench201Model.dataset_version == dataset_version,
    )
    try:
        query = f"{dataset_version}/{framework}/{type}/{input_dim}/{name}"
        model = await session.exec(statement)
        model = model.one_or_none()
    except MultipleResultsFound:
        raise RuntimeError(f"Unexpected Error: Multiple models found for {query}")

    if not model:
        query = f"{dataset_version}/{framework}/{type}/{input_dim}/{name}"
        raise HTTPException(status_code=404,
                            detail=f"Model {query} not found")
    
    return model

def package_model_for_serving(model_path: str) -> FileResponse:
    model_basename = os.path.basename(model_path)
    file = FileResponse(model_path, filename=model_basename)
    content_disposition = f'attachment; filename="{model_basename}"'
    file.headers["Content-Disposition"] = content_disposition

    return file

@app.get("/models/file/{model_id}", response_class=FileResponse)
async def download_model_by_id(model_id: Annotated[int, Path(ge=0)]):
    async with db.get_session(async_engine) as session:
        model = await query_model_by_id(session, model_id)
        file = package_model_for_serving(model.path)
        return file

@app.get("/models/file/{dataset_version}/{framework}/{type}/{input_dim}/{name}",
         response_class=FileResponse)
async def download_model(dataset_version: float, framework: str,
                     input_dim: int, name: str,
                     type: str = Path(regex="^(cells|ops)$"),
                     device: str = "any"):
    async with db.get_session(async_engine) as session:
        model = await query_model_by_index(session, dataset_version, framework,
                                           input_dim, name, type, device)
        file = package_model_for_serving(model.path)
        return file

@app.get("/models/{model_id}", response_model=NASBench201ModelBase)
async def read_model_by_id(model_id: Annotated[int, Path(ge=0)]):
    async with db.get_session(async_engine) as session:
        model = await query_model_by_id(session, model_id)

        return NASBench201ModelBase.from_orm(model)

@app.get("/models/{dataset_version}/{framework}/{type}/{input_dim}/{name}",
         response_model=NASBench201ModelBase)
async def read_model(dataset_version: float, framework: str,
                     input_dim: int, name: str,
                     type: str = Path(regex="^(cells|ops)$"),
                     device: str = "any"):
    async with db.get_session(async_engine) as session:
        model = await query_model_by_index(session, dataset_version, framework,
                                           input_dim, name, type, device)
        return NASBench201ModelBase.from_orm(model)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

