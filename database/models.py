from sqlalchemy import Column, Integer, Float
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Type2Diabetes(Base):
    __tablename__ = "type_2_diabetes"

    id = Column(Integer, primary_key=True, autoincrement=True)

    no_of_pregnancy = Column("No. of Pregnancy", Integer, nullable=False)
    age = Column("Age", Integer, nullable=False)
    bmi = Column("BMI", Float, nullable=False)
    bp_systolic = Column("BP(Systolic)", Integer, nullable=False)
    bp_diastolic = Column("BP(Diastolic)", Float, nullable=True)   # has 4 nulls
    diabetes_pedigree_function = Column("DiabetesPedigreeFunction", Integer, nullable=False)
    insulin = Column("Insulin", Integer, nullable=False)
    skin_thickness_mm = Column("Skin Thickness(mm)", Float, nullable=True)  # has 7 nulls
    type_2_diabetic = Column("Type-2 Diabetic", Integer, nullable=False)
    glucose = Column("Glucose", Float, nullable=False)

def create_tables(engine):
    """creating all this tables in the database"""
    Base.metadata.create_all(engine)
    print("[db] Tables created")