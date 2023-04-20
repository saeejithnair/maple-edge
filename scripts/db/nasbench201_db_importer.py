import os
import asyncio
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi import Depends

from maple.db.database import create_db_and_tables, get_session
from maple.db.models import NASBench201Model

DATASET_VERSION=1.0

async def create_model(*, session: AsyncSession = Depends(get_session),
                 model_name: str, framework: str, dataset_version: float,
                 model_type: str, device: str, path: str):
    model = NASBench201Model(
        name=model_name,
        framework=framework,
        dataset_version=dataset_version,
        model_type=model_type,
        device=device,
        path=path
    )
    session.add(model)
    await session.commit()

async def import_models(*, session: AsyncSession = Depends(get_session),
                        root_dir: str):
    for subdir, dirs, files in os.walk(root_dir):
        if 'cells' in subdir or 'ops' in subdir:
            parts = subdir.split('models/')[1].split('/')
            framework = parts[1]
            model_type = parts[-1]

            if framework == 'trt':
                device = parts[2]
            else:
                device = None

            for file in files:
                model_name = file
                path = os.path.join(subdir, file)

                await create_model(session, model_name, framework,
                                   DATASET_VERSION, model_type, device, path)

async def main():
    ROOT_DIR = '/pub4/smnair/cheetah/models'

    await create_db_and_tables()
    await import_models(root_dir=ROOT_DIR)

if __name__ == '__main__':
    asyncio.run(main())
