"""Rebuild coordinate-derived fields without changing identities or results."""

import json
import sqlite3
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path

from rdkit import Chem, rdBase

from chemsmart.database.utils import check_schema_version, from_json, to_json
from chemsmart.io.molecules.structure import Molecule

STRUCTURE_FIELDS = (
    "smiles",
    "inchi",
    "chiral_centers_json",
    "is_chiral",
    "is_ring",
    "is_aromatic",
    "is_multicomponent",
    "num_components",
)
CONVERSION_TABLE = """
CREATE TABLE IF NOT EXISTS structure_conversions (
    conversion_index INTEGER PRIMARY KEY,
    structure_id TEXT,
    molecule_id TEXT,
    created_at TEXT NOT NULL,
    method TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    old_values_json TEXT,
    new_values_json TEXT
)
"""
CONVERSION_METHOD = f"rdkit-{rdBase.rdkitVersion}/structure-fields-v1"


class StructureRepairer:
    """Inspect every stored geometry before updating shared molecular fields."""

    def __init__(self, db_file):
        self.db_file = Path(db_file).resolve()
        if not self.db_file.is_file():
            raise ValueError(f"Database does not exist: {self.db_file}")
        check_schema_version(str(self.db_file))

    def run(self, report, apply=False, scope="all", limit=None):
        """Stream an exclusive JSONL report; writes require explicit apply.

        Limit counts molecular species, not geometries. Every geometry belonging
        to a selected species is checked. Failed or conflicting groups stay intact.
        Updates and their audit rows commit together, per species.
        """
        if scope not in {"all", "invalid", "missing"}:
            raise ValueError("scope must be all, invalid, or missing")
        if limit is not None and limit < 1:
            raise ValueError("limit must be positive")
        report = Path(report).resolve()
        if report == self.db_file:
            raise ValueError("Report cannot overwrite the database")
        summary = dict(
            checked=0, unchanged=0, repairable=0, failed=0, updated=0
        )
        # Read-only mode also prevents dry runs from creating SQLite side tables.
        mode = "rw" if apply else "ro"
        conn = sqlite3.connect(
            f"{self.db_file.as_uri()}?mode={mode}", uri=True
        )
        conn.row_factory = sqlite3.Row
        try:
            with report.open("x", encoding="utf-8") as output:
                if apply:
                    conn.execute(CONVERSION_TABLE)
                    conn.commit()
                columns = ", ".join(f"m.{field}" for field in STRUCTURE_FIELDS)
                rows = conn.execute(f"""
                    SELECT m.molecule_id, {columns}, s.structure_id, s.charge,
                           s.multiplicity, s.chemical_symbols_json, s.positions_json
                    FROM molecules m LEFT JOIN structures s
                    ON s.molecule_id = m.molecule_id
                    ORDER BY m.molecule_id, s.structure_id
                """)
                for molecule_id, group in groupby(
                    rows, lambda row: row["molecule_id"]
                ):
                    geometries = list(group)
                    old = {
                        field: geometries[0][field]
                        for field in STRUCTURE_FIELDS
                    }
                    if scope == "missing" and old["smiles"]:
                        continue
                    if scope == "invalid" and (
                        not old["smiles"]
                        or Chem.MolFromSmiles(old["smiles"]) is not None
                    ):
                        continue
                    if limit is not None and summary["checked"] >= limit:
                        break
                    summary["checked"] += 1
                    entry = {
                        "molecule_id": molecule_id,
                        "method": CONVERSION_METHOD,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "old": old,
                        "structures": [],
                        "new": None,
                        "status": "failed",
                        "error": None,
                    }
                    candidates = []
                    for row in geometries:
                        detail = {"structure_id": row["structure_id"]}
                        try:
                            mol = Molecule(
                                symbols=from_json(
                                    row["chemical_symbols_json"]
                                ),
                                positions=from_json(row["positions_json"]),
                                charge=row["charge"],
                                multiplicity=row["multiplicity"],
                            )
                            fields = mol.get_structure_fields()
                            fields["chiral_centers_json"] = to_json(
                                fields.pop("chiral_centers")
                            )
                            detail.update(status="valid", fields=fields)
                            candidates.append(fields)
                        except (ValueError, RuntimeError, TypeError) as exc:
                            detail.update(status="failed", error=str(exc))
                        entry["structures"].append(detail)
                    if len(candidates) != len(geometries):
                        entry["error"] = (
                            "One or more geometries failed conversion"
                        )
                    elif any(
                        candidate != candidates[0]
                        for candidate in candidates[1:]
                    ):
                        entry["error"] = (
                            "Geometries disagree on shared fields; manual review required"
                        )
                    else:
                        entry["new"] = candidates[0]
                        entry["status"] = (
                            "unchanged"
                            if old == candidates[0]
                            else "repairable"
                        )
                    summary[entry["status"]] += 1
                    if apply:
                        with conn:
                            if entry["status"] == "repairable":
                                assignments = ", ".join(
                                    f"{field} = ?"
                                    for field in STRUCTURE_FIELDS
                                )
                                conn.execute(
                                    f"UPDATE molecules SET {assignments} WHERE molecule_id = ?",
                                    [
                                        entry["new"][field]
                                        for field in STRUCTURE_FIELDS
                                    ]
                                    + [molecule_id],
                                )
                                summary["updated"] += 1
                            for detail in entry["structures"]:
                                conn.execute(
                                    """INSERT INTO structure_conversions
                                    (structure_id, molecule_id, created_at, method, status,
                                     error, old_values_json, new_values_json)
                                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                                    (
                                        detail["structure_id"],
                                        molecule_id,
                                        entry["created_at"],
                                        CONVERSION_METHOD,
                                        entry["status"],
                                        detail.get("error", entry["error"]),
                                        to_json(old),
                                        to_json(entry["new"]),
                                    ),
                                )
                    entry["applied"] = (
                        apply and entry["status"] == "repairable"
                    )
                    output.write(json.dumps(entry, ensure_ascii=False) + "\n")
                    output.flush()
        finally:
            conn.close()
        return summary
