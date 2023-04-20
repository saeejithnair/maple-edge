import os
import asyncio
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi import Depends

from maple.db import models
from maple.db.models import NASBench201Model
from maple.db.database import create_db_and_tables, get_session

DATASET_VERSION=1.0

async def create_model(*, session: AsyncSession, model_name: str,
                       framework: str, dataset_version: float,
                       model_type: str, device: str, input_dim: int,
                       arch_idx: int, path: str):
    model = NASBench201Model(
        name=model_name,
        framework=framework,
        dataset_version=dataset_version,
        model_type=model_type,
        device=device,
        path=path,
        input_dim=input_dim,
        arch_idx=arch_idx,
        size_bytes=os.path.getsize(path)
    )
    session.add(model)
    await session.commit()

async def import_models(*, session: AsyncSession,
                        root_dir: str):
    for subdir, dirs, files in os.walk(root_dir):
        if 'cells' in subdir or 'ops' in subdir:
            parts = subdir.split('models/')[1].split('/')
            input_dim = int(parts[0].split('_')[1])
            framework = parts[1]
            model_type = parts[-1]

            if framework == 'trt':
                device = parts[2]
            else:
                device = "any"

            for file in files:
                model_name = os.path.splitext(file)[0]
                path = os.path.join(subdir, file)

                if "cells" in subdir:
                    arch_idx = int(model_name.split('_')[-1])
                else:
                    arch_idx = -1

                await create_model(session=session,
                                   model_name=model_name,
                                   framework=framework,
                                   dataset_version=DATASET_VERSION,
                                   model_type=model_type,
                                   device=device,
                                   input_dim=input_dim,
                                   arch_idx=arch_idx,
                                   path=path)

async def main():
    ROOT_DIR = '/pub4/smnair/cheetah/models'

    await create_db_and_tables()
    async with get_session() as session:
        await import_models(session=session, root_dir=ROOT_DIR)

if __name__ == '__main__':
    asyncio.run(main())
