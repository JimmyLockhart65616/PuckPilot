from puckpilot.data import store


def test_init_db_creates_tables(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_db(conn)
    tables = store.table_names(conn)
    assert {"sync_meta", "nhl_players", "nhl_schedule", "nhl_game_logs"} <= tables


def test_init_db_is_idempotent(tmp_path):
    conn = store.connect(tmp_path / "t.db")
    store.init_db(conn)
    conn.execute("INSERT INTO sync_meta (key, value) VALUES ('a', '1')")
    conn.commit()
    store.init_db(conn)  # must not wipe data
    row = conn.execute("SELECT value FROM sync_meta WHERE key='a'").fetchone()
    assert row["value"] == "1"


def test_connect_creates_parent_dirs(tmp_path):
    conn = store.connect(tmp_path / "nested" / "dirs" / "t.db")
    store.init_db(conn)
    assert (tmp_path / "nested" / "dirs" / "t.db").exists()
