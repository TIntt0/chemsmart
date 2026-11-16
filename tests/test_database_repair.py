"""Scientific regression and persistence checks for structure repair."""

import json
import sqlite3

import pytest
from click.testing import CliRunner
from rdkit import Chem

from chemsmart.cli.database import database
from chemsmart.database.assemble import BaseAssembler
from chemsmart.database.database import Database
from chemsmart.database.repair import StructureRepairer
from chemsmart.database.utils import to_json
from chemsmart.io.molecules.structure import Molecule
from tests.helpers import rdkit_embed_and_optimize


@pytest.mark.parametrize(
    "smiles", ["N", "O", "C#C", "CO", "c1ccccc1", "[NH4+]", "C[C@H](O)F"]
)
def test_shared_fields_match_reference(smiles):
    reference = rdkit_embed_and_optimize(smiles)
    mol = Molecule(
        symbols=[atom.GetSymbol() for atom in reference.GetAtoms()],
        positions=reference.GetConformer().GetPositions(),
        charge=Chem.GetFormalCharge(reference),
        multiplicity=1,
    )
    fields = mol.get_structure_fields()
    parsed = Chem.MolFromSmiles(fields["smiles"])
    assert Chem.MolToSmiles(parsed) == Chem.MolToSmiles(
        Chem.MolFromSmiles(smiles)
    )
    assert fields["inchi"] == Chem.MolToInchi(reference)
    assert fields["is_aromatic"] == any(
        a.GetIsAromatic() for a in reference.GetAtoms()
    )
    assert fields["is_ring"] == bool(Chem.GetSymmSSSR(reference))
    assert fields["num_components"] == 1
    assert fields["is_multicomponent"] is False
    assert fields["chiral_centers"] == {
        idx + 1: label
        for idx, label in Chem.FindMolChiralCenters(
            reference, force=True, includeUnassigned=True
        )
    }


def test_dry_run_apply_and_idempotency(repair_database, tmp_path):
    repairer = StructureRepairer(repair_database)
    original = repair_database.read_bytes()
    report = tmp_path / "preview.jsonl"
    summary = repairer.run(report)
    assert summary["repairable"] == 1
    assert summary["updated"] == 0
    assert repair_database.read_bytes() == original
    entry = json.loads(report.read_text())
    assert entry["old"]["smiles"] == "[H]o[H]"
    assert entry["new"]["is_aromatic"] is False
    assert not entry["applied"]
    summary = repairer.run(tmp_path / "apply.jsonl", apply=True)
    assert summary["updated"] == 1
    with sqlite3.connect(repair_database) as conn:
        smiles, aromatic = conn.execute(
            "SELECT smiles, is_aromatic FROM molecules"
        ).fetchone()
        assert Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) == "O"
        assert aromatic == 0
        assert (
            conn.execute("SELECT energy FROM record_structures").fetchone()[0]
            == -5.0
        )
        assert (
            conn.execute("SELECT total_energy FROM records").fetchone()[0]
            == -5.0
        )
        assert conn.execute(
            "SELECT molecule_id, structure_id FROM structures"
        ).fetchone() == ("water", "water-1")
        audit = conn.execute(
            "SELECT old_values_json FROM structure_conversions"
        ).fetchone()[0]
        assert json.loads(audit)["smiles"] == "[H]o[H]"
    summary = repairer.run(tmp_path / "again.jsonl", apply=True)
    assert summary["unchanged"] == 1
    assert summary["updated"] == 0
    with pytest.raises(FileExistsError):
        repairer.run(report, apply=True)


@pytest.mark.parametrize(
    "scope, expected", [("all", 1), ("invalid", 1), ("missing", 0)]
)
def test_repair_selection(repair_database, tmp_path, scope, expected):
    assert (
        StructureRepairer(repair_database).run(
            tmp_path / "report.jsonl", scope=scope, limit=1
        )["checked"]
        == expected
    )


def test_failed_geometry_is_preserved(repair_database, tmp_path):
    with sqlite3.connect(repair_database) as conn:
        conn.execute(
            "UPDATE structures SET positions_json = ?", (to_json([[0, 0, 0]]),)
        )
    summary = StructureRepairer(repair_database).run(
        tmp_path / "failed.jsonl", apply=True
    )
    assert summary["failed"] == 1
    assert summary["updated"] == 0
    with sqlite3.connect(repair_database) as conn:
        assert (
            conn.execute("SELECT smiles FROM molecules").fetchone()[0]
            == "[H]o[H]"
        )


def test_conflicting_geometries_are_preserved(repair_database, tmp_path):
    with sqlite3.connect(repair_database) as conn:
        conn.execute(
            """INSERT INTO structures (structure_id, molecule_id, charge,
            multiplicity, chemical_symbols_json, positions_json)
            VALUES ('other', 'water', 0, 1, ?, ?)""",
            (
                to_json(["N", "H", "H", "H"]),
                to_json(
                    [
                        [0, 0, 0],
                        [0.94, 0, 0.38],
                        [-0.47, 0.81, 0.38],
                        [-0.47, -0.81, 0.38],
                    ]
                ),
            ),
        )
    summary = StructureRepairer(repair_database).run(
        tmp_path / "conflict.jsonl", apply=True
    )
    assert summary["failed"] == 1
    entry = json.loads((tmp_path / "conflict.jsonl").read_text())
    assert "disagree" in entry["error"]
    assert summary["updated"] == 0


def test_repair_cli_defaults_to_preview(repair_database, tmp_path):
    result = CliRunner().invoke(
        database,
        [
            "repair-structure",
            "-f",
            str(repair_database),
            "--report",
            str(tmp_path / "cli.jsonl"),
        ],
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["updated"] == 0


def test_assembly_keeps_failed_conversion(
    tmp_path, water_molecule, monkeypatch
):
    monkeypatch.setattr(
        Molecule,
        "get_structure_fields",
        lambda self: (_ for _ in ()).throw(ValueError("invalid graph")),
    )
    info = BaseAssembler.__new__(BaseAssembler).get_molecule_info(
        water_molecule
    )
    assert info["smiles"] is None
    assert info["is_aromatic"] is None
    assert info["structure_conversion"]["status"] == "failed"
    assert info["positions"] is not None
    db = Database(str(tmp_path / "failed.db"))
    db.create()
    info["index"] = 1
    db.insert_record({"record_id": "failed", "molecules": [info]})
    with sqlite3.connect(db.db_file) as conn:
        assert conn.execute(
            "SELECT smiles, is_aromatic FROM molecules"
        ).fetchone() == (None, None)
        assert conn.execute(
            "SELECT status, error FROM structure_conversions"
        ).fetchone() == ("failed", "invalid graph")


def test_fragment_fields_use_validated_connectivity():
    mol = Molecule(
        symbols=["O", "H", "H", "O", "H", "H"],
        positions=[
            [0, 0, 0],
            [0.96, 0, 0],
            [-0.24, 0.93, 0],
            [10, 0, 0],
            [10.96, 0, 0],
            [9.76, 0.93, 0],
        ],
        charge=0,
        multiplicity=1,
    )
    fields = mol.get_structure_fields()
    assert fields["num_components"] == 2
    assert fields["is_multicomponent"] is True
    assert fields["is_aromatic"] is False
    assert fields["is_ring"] is False


def test_assembly_uses_shared_graph(water_molecule):
    info = BaseAssembler.__new__(BaseAssembler).get_molecule_info(
        water_molecule
    )
    for field, value in water_molecule.get_structure_fields().items():
        assert info[field] == value
    assert info["structure_conversion"] == {"status": "valid", "error": None}


def test_limit_and_multiple_species(repair_database, tmp_path):
    with sqlite3.connect(repair_database) as conn:
        conn.execute(
            "INSERT INTO molecules (molecule_id, smiles) VALUES ('water-b', '')"
        )
        conn.execute("""INSERT INTO structures
            (structure_id, molecule_id, charge, multiplicity, chemical_symbols_json, positions_json)
            SELECT 'water-2', 'water-b', charge, multiplicity, chemical_symbols_json, positions_json
            FROM structures WHERE structure_id = 'water-1'""")
        original_structures = conn.execute(
            "SELECT * FROM structures ORDER BY structure_id"
        ).fetchall()
    repairer = StructureRepairer(repair_database)
    summary = repairer.run(tmp_path / "limited.jsonl", limit=1)
    assert summary["checked"] == 1
    summary = repairer.run(tmp_path / "missing.jsonl", scope="missing")
    assert summary["checked"] == 1
    assert (
        json.loads((tmp_path / "missing.jsonl").read_text())["molecule_id"]
        == "water-b"
    )
    summary = repairer.run(tmp_path / "both.jsonl", apply=True)
    assert summary["updated"] == 2
    with sqlite3.connect(repair_database) as conn:
        assert (
            conn.execute(
                "SELECT * FROM structures ORDER BY structure_id"
            ).fetchall()
            == original_structures
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM structure_conversions"
            ).fetchone()[0]
            == 2
        )
