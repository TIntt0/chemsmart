"""Database fixtures."""

import os

import pytest


############ Database Fixtures ##################
@pytest.fixture()
def database_test_directory(test_data_directory):
    return os.path.join(test_data_directory, "DatabaseTests")


@pytest.fixture()
def database_chemsmart_file(database_test_directory):
    return os.path.join(database_test_directory, "chemsmart.db")


@pytest.fixture()
def database_chemsmart_xtb_file(database_test_directory):
    return os.path.join(database_test_directory, "chemsmart_xtb.db")


@pytest.fixture()
def database_ase_file(database_test_directory):
    return os.path.join(database_test_directory, "ase.db")


@pytest.fixture()
def database_empty_file(database_test_directory):
    return os.path.join(database_test_directory, "empty.db")


@pytest.fixture()
def repair_database(tmp_path):
    """Legacy water record with invalid SMILES and intact calculation data."""
    import sqlite3

    from chemsmart.database.database import Database
    from chemsmart.database.utils import to_json

    path = tmp_path / "legacy.db"
    Database(str(path)).create()
    with sqlite3.connect(path) as conn:
        conn.execute(
            "INSERT INTO molecules (molecule_id, smiles, is_aromatic) VALUES (?, ?, ?)",
            ("water", "[H]o[H]", 1),
        )
        conn.execute(
            """INSERT INTO structures (structure_id, molecule_id, charge,
            multiplicity, chemical_symbols_json, positions_json)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                "water-1",
                "water",
                0,
                1,
                to_json(["O", "H", "H"]),
                to_json([[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]]),
            ),
        )
        conn.execute(
            "INSERT INTO records (record_id, total_energy) VALUES ('r1', -5.0)"
        )
        conn.execute(
            "INSERT INTO record_structures (record_id, structure_id, index_in_record, energy) VALUES ('r1', 'water-1', 1, -5.0)"
        )
    return path
