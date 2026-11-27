"""Variable metadata: the single source of truth for labels, types and roles.

Drives the data dictionary (Description tab) *and* which variables the Group
summary tab can describe, so adding a column is a one-line change here.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

RCOUNT_LEVELS = ["0", "1", "2", "3", "4", "5+"]


@dataclass(frozen=True)
class Variable:
    name: str
    label: str
    kind: str          # continuous | categorical | binary | id | date
    description: str
    unit: str = ""
    role: str = "feature"   # feature | target | identifier | date | derived


VARIABLES: list[Variable] = [
    Variable("eid", "Episode ID", "id", "Unique episode (admission) identifier", role="identifier"),
    Variable("vdate", "Visit date", "date", "Date of admission", role="date"),
    Variable("discharged", "Discharge date", "date",
             "Date of discharge — a near-perfect proxy for the target, exclude when predicting", role="date"),
    Variable("lengthofstay", "Length of stay", "continuous", "Days spent in hospital", "days", "target"),
    Variable("rcount", "Readmission count", "categorical", "Number of readmissions ('5+' = five or more)"),
    Variable("gender", "Gender", "categorical", "Patient gender (M / F)"),
    Variable("facid", "Facility", "categorical", "Facility identifier"),
    Variable("dialysisrenalendstage", "Dialysis / end-stage renal", "binary", "Dialysis indicator"),
    Variable("asthma", "Asthma", "binary", "Asthma indicator"),
    Variable("irondef", "Iron deficiency", "binary", "Iron deficiency indicator"),
    Variable("pneum", "Pneumonia", "binary", "Pneumonia indicator"),
    Variable("substancedependence", "Substance dependence", "binary", "Substance dependency indicator"),
    Variable("psychologicaldisordermajor", "Major psychological disorder", "binary",
             "Major psychological disorder indicator"),
    Variable("depress", "Depression", "binary", "Depression indicator"),
    Variable("psychother", "Other psychological disorder", "binary", "Other psychological disorders indicator"),
    Variable("fibrosisandother", "Fibrosis and other", "binary", "Fibrosis indicator"),
    Variable("malnutrition", "Malnutrition", "binary", "Malnutrition indicator"),
    Variable("hemo", "Hemo", "binary", "Hemo indicator"),
    Variable("hematocrit", "Hematocrit", "continuous", "Hematocrit level", "g/dL"),
    Variable("neutrophils", "Neutrophils", "continuous", "White blood cell (neutrophil) count", "cells/µL"),
    Variable("sodium", "Sodium", "continuous", "Sodium level", "mmol/L"),
    Variable("glucose", "Glucose", "continuous", "Glucose level", "mmol/L"),
    Variable("bloodureanitro", "Blood urea nitrogen", "continuous", "Blood urea nitrogen level", "mg/dL"),
    Variable("creatinine", "Creatinine", "continuous", "Creatinine level", "mg/dL"),
    Variable("bmi", "BMI", "continuous", "Body mass index", "kg/m²"),
    Variable("pulse", "Pulse", "continuous", "Pulse rate", "beats/min"),
    Variable("respiration", "Respiration", "continuous", "Respiration rate", "breaths/min"),
    #Variable("secondarydiagnosisnonicd9", "Secondary diagnoses", "continuous",
    #         "Number of secondary (non-ICD-9) diagnoses", "count"),
    Variable("comorbidity_count", "Comorbidity count", "continuous",
             "Derived: sum of the 11 binary comorbidity flags", "count", "derived"),
]

LABELS = {v.name: v.label for v in VARIABLES}


def display_label(v: Variable) -> str:
    """Label as shown in the summary tables: 'Glucose (mmol/L)'. A unit of 'count' is a
    description of the quantity, not a unit, so those variables keep their plain label."""
    return f"{v.label} ({v.unit})" if v.unit and v.unit != "count" else v.label


DISPLAY_LABELS = {v.name: display_label(v) for v in VARIABLES}


def dictionary_frame() -> pd.DataFrame:
    """The data dictionary as a display table (one row per variable)."""
    return pd.DataFrame([
        {"Variable": v.label, "Column": v.name, "Type": v.kind, "Role": v.role,
         "Unit": v.unit, "Description": v.description}
        for v in VARIABLES
    ])


def names_of(kind: str) -> list[str]:
    """Variable names of one kind, in dictionary order."""
    return [v.name for v in VARIABLES if v.kind == kind]


def choices_of(*kinds: str) -> dict[str, str]:
    """{name: label} for `ui.input_select/selectize` of the given kinds."""
    return {v.name: v.label for v in VARIABLES if v.kind in kinds}


# Grouping options for the Group summary tab (value -> label).
GROUP_BY_CHOICES = {
    "none": "All entries",
    #"facid": "Facility",
    #"gender": "Gender",
    #"rcount": "Readmission count",
    "split": "Train / test split",
}
