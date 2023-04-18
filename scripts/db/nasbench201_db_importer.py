import os
import re
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from maple.db.database import database, engine, SessionLocal
Base = declarative_base()

DATASET_VERSION=1
class NASBench201Model(Base):
    __tablename__ = 'NASBench201_Models'
    id = Column(Integer, primary_key=True)
    model_name = Column(String)
    framework = Column(String)
    dataset_version = Column(Integer)
    model_type = Column(String)
    device = Column(String)
    path = Column(String)

def add_model_to_db(session, model_name, framework, dataset_version, model_type, device, path):
    new_model = NASBench201Model(
        model_name=model_name,
        framework=framework,
        dataset_version=dataset_version,
        model_type=model_type,
        device=device,
        path=path
    )
    session.add(new_model)
    session.commit()

def import_models(root_dir, session):
    for subdir, dirs, files in os.walk(root_dir):
        if 'cells' in subdir or 'ops' in subdir:
            parts = subdir.split('models/')[1].split('/')
            framework = parts[1]
            model_type = parts[-1]

            if framework == 'trt':
                device = parts[2]
            else:
                device = "generic"

            for file in files:
                model_name = file
                path = os.path.join(subdir, file)

                add_model_to_db(session, model_name, framework, DATASET_VERSION, model_type, device, path)

if __name__ == '__main__':
    ROOT_DIR = '/pub4/smnair/cheetah/models'

    # NASBench201Model.metadata.create_all(engine)

    # session = SessionLocal()
    session = None
    import_models(ROOT_DIR, session)

    # models = session.query(NASBench201Model).all()
    # for model in models:
    #     print(model.model_name, model.framework, model.dataset_version, model.model_type, model.device, model.path)

    # session.close()