-- 0001_initial: core tables for templates, garments, compatibility and jobs.
--
-- Design notes:
--  * Records are stored as strict JSON in a `payload` column plus a small set of
--    promoted columns for indexing/filtering. The Pydantic model remains the
--    single source of truth for shape; SQLite is a durable index over it.
--  * (id, version) is the primary key for versioned entities. A new version is
--    a new row; existing rows are never rewritten, which is what makes a job's
--    recorded template_version meaningful.

CREATE TABLE human_templates (
    id                      TEXT    NOT NULL,
    version                 INTEGER NOT NULL,
    display_name            TEXT    NOT NULL,
    source_sha256           TEXT    NOT NULL,
    width                   INTEGER NOT NULL,
    height                  INTEGER NOT NULL,
    fps                     REAL    NOT NULL,
    frame_count             INTEGER NOT NULL,
    transition_anchor_frame INTEGER NOT NULL,
    template_clothing_class TEXT    NOT NULL,
    status                  TEXT    NOT NULL,
    created_at              TEXT    NOT NULL,
    updated_at              TEXT,
    payload                 TEXT    NOT NULL,
    PRIMARY KEY (id, version)
);
CREATE INDEX idx_templates_status ON human_templates (status);
CREATE INDEX idx_templates_source ON human_templates (source_sha256);

CREATE TABLE garment_assets (
    id            TEXT    NOT NULL,
    version       INTEGER NOT NULL,
    product_name  TEXT,
    brand         TEXT,
    category      TEXT    NOT NULL,
    body_coverage TEXT    NOT NULL,
    status        TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT,
    payload       TEXT    NOT NULL,
    PRIMARY KEY (id, version)
);
CREATE INDEX idx_garments_status ON garment_assets (status);
CREATE INDEX idx_garments_category ON garment_assets (category);

CREATE TABLE compatibility_reports (
    id               TEXT    PRIMARY KEY,
    template_id      TEXT    NOT NULL,
    template_version INTEGER NOT NULL,
    garment_id       TEXT    NOT NULL,
    garment_version  INTEGER NOT NULL,
    state            TEXT    NOT NULL,
    rules_version    TEXT    NOT NULL,
    confidence       REAL    NOT NULL,
    has_override     INTEGER NOT NULL DEFAULT 0,
    created_at       TEXT    NOT NULL,
    updated_at       TEXT,
    payload          TEXT    NOT NULL,
    FOREIGN KEY (template_id, template_version)
        REFERENCES human_templates (id, version) ON DELETE RESTRICT,
    FOREIGN KEY (garment_id, garment_version)
        REFERENCES garment_assets (id, version) ON DELETE RESTRICT
);
CREATE INDEX idx_reports_pair ON compatibility_reports (
    template_id, template_version, garment_id, garment_version
);
CREATE INDEX idx_reports_state ON compatibility_reports (state);

CREATE TABLE render_jobs (
    id                       TEXT    PRIMARY KEY,
    template_id              TEXT    NOT NULL,
    template_version         INTEGER NOT NULL,
    garment_id               TEXT    NOT NULL,
    garment_version          INTEGER NOT NULL,
    compatibility_report_id  TEXT,
    backend_name             TEXT    NOT NULL,
    backend_version          TEXT    NOT NULL,
    workflow_id              TEXT,
    workflow_sha256          TEXT,
    seed                     INTEGER NOT NULL,
    frame_start              INTEGER NOT NULL,
    frame_end                INTEGER NOT NULL,
    status                   TEXT    NOT NULL,
    completed_frames         INTEGER NOT NULL DEFAULT 0,
    total_frames             INTEGER NOT NULL DEFAULT 0,
    created_at               TEXT    NOT NULL,
    updated_at               TEXT,
    started_at               TEXT,
    finished_at              TEXT,
    payload                  TEXT    NOT NULL,
    FOREIGN KEY (template_id, template_version)
        REFERENCES human_templates (id, version) ON DELETE RESTRICT,
    FOREIGN KEY (garment_id, garment_version)
        REFERENCES garment_assets (id, version) ON DELETE RESTRICT,
    FOREIGN KEY (compatibility_report_id)
        REFERENCES compatibility_reports (id) ON DELETE RESTRICT
);
CREATE INDEX idx_jobs_status ON render_jobs (status);
CREATE INDEX idx_jobs_template ON render_jobs (template_id, template_version);
CREATE INDEX idx_jobs_garment ON render_jobs (garment_id, garment_version);

-- Append-only audit log. Never updated, never deleted by application code.
CREATE TABLE audit_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    occurred_at TEXT NOT NULL,
    actor       TEXT,
    event       TEXT NOT NULL,
    entity_type TEXT,
    entity_id   TEXT,
    details     TEXT
);
CREATE INDEX idx_audit_entity ON audit_events (entity_type, entity_id);
CREATE INDEX idx_audit_event ON audit_events (event);
