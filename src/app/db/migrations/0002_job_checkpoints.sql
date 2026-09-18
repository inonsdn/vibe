-- 0002_job_checkpoints: per-frame completion records driving resume.
--
-- The checkpoint inside the job payload is a summary; this table is the
-- authoritative, crash-safe record of which frames are finished. A frame row
-- is written only after its composited PNG is on disk, so a resumed job can
-- trust it without re-reading every file.

CREATE TABLE job_frames (
    job_id        TEXT    NOT NULL,
    frame_index   INTEGER NOT NULL,
    status        TEXT    NOT NULL,          -- rendered | composited | failed
    frame_sha256  TEXT,
    frame_seed    INTEGER,
    duration_ms   INTEGER,
    attempts      INTEGER NOT NULL DEFAULT 1,
    error         TEXT,
    updated_at    TEXT    NOT NULL,
    PRIMARY KEY (job_id, frame_index),
    FOREIGN KEY (job_id) REFERENCES render_jobs (id) ON DELETE CASCADE
);
CREATE INDEX idx_job_frames_status ON job_frames (job_id, status);

CREATE TABLE job_manifests (
    job_id      TEXT PRIMARY KEY,
    digest      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    payload     TEXT NOT NULL,
    FOREIGN KEY (job_id) REFERENCES render_jobs (id) ON DELETE CASCADE
);
CREATE INDEX idx_job_manifests_digest ON job_manifests (digest);
