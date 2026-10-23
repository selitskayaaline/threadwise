CREATE TABLE IF NOT EXISTS channel_access (
    user_name text NOT NULL,
    channel   text NOT NULL,
    PRIMARY KEY (user_name, channel)
);

INSERT INTO channel_access (user_name, channel) VALUES
    ('lead', 'backend'),
    ('lead', 'devops'),
    ('lead', 'frontend'),
    ('lead', 'product'),
    ('lead', 'general'),
    ('frontend-dev', 'frontend'),
    ('frontend-dev', 'product'),
    ('frontend-dev', 'general'),
    ('guest', 'general')
ON CONFLICT DO NOTHING;
