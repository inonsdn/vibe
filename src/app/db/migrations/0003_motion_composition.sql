-- 0003_motion_composition: motion references, compositions and master candidates.
--
-- The Motion Composition phase produces artifacts that precede a Master Human
-- Performance. They are versioned the same way as templates and garments: a new
-- version is a new row, so a composition's recorded source versions stay
-- meaningful forever.
--
-- Note what is NOT here: nothing stores imagery from a motion reference. A
-- motion source row points at the operator's own file and at a directory of
-- pose JSON. That separation is the product requirement.

CREATE TABLE motion_sources (
    id              TEXT    NOT NULL,
    version         INTEGER NOT NULL,
    display_name    TEXT    NOT NULL,
    source_sha256   TEXT    NOT NULL,
    width           INTEGER NOT NULL,
    height          INTEGER NOT NULL,
    fps             REAL    NOT NULL,
    frame_count     INTEGER NOT NULL,
    range_start     INTEGER NOT NULL,
    range_end       INTEGER NOT NULL,
    pose_format     TEXT    NOT NULL,
    pose_origin     TEXT    NOT NULL,
    status          TEXT    NOT NULL,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT,
    payload         TEXT    NOT NULL,
    PRIMARY KEY (id, version)
);
CREATE INDEX idx_motion_sources_status ON motion_sources (status);
CREATE INDEX idx_motion_sources_sha ON motion_sources (source_sha256);

CREATE TABLE skeleton_profiles (
    id          TEXT    NOT NULL,
    version     INTEGER NOT NULL,
    created_at  TEXT    NOT NULL,
    updated_at  TEXT,
    payload     TEXT    NOT NULL,
    PRIMARY KEY (id, version)
);

CREATE TABLE hero_characters (
    id            TEXT    NOT NULL,
    version       INTEGER NOT NULL,
    display_name  TEXT    NOT NULL,
    subject_kind  TEXT    NOT NULL,
    created_at    TEXT    NOT NULL,
    updated_at    TEXT,
    payload       TEXT    NOT NULL,
    PRIMARY KEY (id, version)
);

CREATE TABLE motion_compositions (
    id                  TEXT    NOT NULL,
    version             INTEGER NOT NULL,
    display_name        TEXT    NOT NULL,
    segment_count       INTEGER NOT NULL,
    join_count          INTEGER NOT NULL,
    output_fps          REAL    NOT NULL,
    output_frame_count  INTEGER NOT NULL,
    profile_id          TEXT    NOT NULL,
    profile_version     INTEGER NOT NULL,
    status              TEXT    NOT NULL,
    created_at          TEXT    NOT NULL,
    updated_at          TEXT,
    payload             TEXT    NOT NULL,
    PRIMARY KEY (id, version),
    FOREIGN KEY (profile_id, profile_version)
        REFERENCES skeleton_profiles (id, version) ON DELETE RESTRICT
);
CREATE INDEX idx_compositions_status ON motion_compositions (status);

CREATE TABLE master_candidates (
    id                    TEXT    PRIMARY KEY,
    version               INTEGER NOT NULL,
    display_name          TEXT    NOT NULL,
    origin                TEXT    NOT NULL,
    composition_id        TEXT    NOT NULL,
    composition_version   INTEGER NOT NULL,
    hero_id               TEXT    NOT NULL,
    hero_version          INTEGER NOT NULL,
    backend_name          TEXT    NOT NULL,
    backend_version       TEXT    NOT NULL,
    seed                  INTEGER NOT NULL,
    frame_count           INTEGER NOT NULL,
    status                TEXT    NOT NULL,
    accepted              INTEGER NOT NULL DEFAULT 0,
    promoted_template_id  TEXT,
    created_at            TEXT    NOT NULL,
    updated_at            TEXT,
    payload               TEXT    NOT NULL,
    FOREIGN KEY (composition_id, composition_version)
        REFERENCES motion_compositions (id, version) ON DELETE RESTRICT,
    FOREIGN KEY (hero_id, hero_version)
        REFERENCES hero_characters (id, version) ON DELETE RESTRICT
);
CREATE INDEX idx_master_candidates_status ON master_candidates (status);
CREATE INDEX idx_master_candidates_accepted ON master_candidates (accepted);

-- Crash-safe per-chunk record, the animation analogue of job_frames. A chunk
-- row is written only after every one of its frames is on disk, so resume can
-- trust it without re-reading the directory.
CREATE TABLE master_chunks (
    candidate_id  TEXT    NOT NULL,
    chunk_index   INTEGER NOT NULL,
    start_frame   INTEGER NOT NULL,
    end_frame     INTEGER NOT NULL,
    status        TEXT    NOT NULL,
    seed          INTEGER,
    overlap_frames INTEGER NOT NULL DEFAULT 0,
    duration_ms   INTEGER,
    attempts      INTEGER NOT NULL DEFAULT 1,
    error         TEXT,
    updated_at    TEXT    NOT NULL,
    PRIMARY KEY (candidate_id, chunk_index),
    FOREIGN KEY (candidate_id) REFERENCES master_candidates (id) ON DELETE CASCADE
);
CREATE INDEX idx_master_chunks_status ON master_chunks (candidate_id, status);

CREATE TABLE master_manifests (
    candidate_id  TEXT PRIMARY KEY,
    digest        TEXT NOT NULL,
    created_at    TEXT NOT NULL,
    payload       TEXT NOT NULL,
    FOREIGN KEY (candidate_id) REFERENCES master_candidates (id) ON DELETE CASCADE
);
