-- Apply explicitly to each PostgreSQL environment before enabling password
-- recovery. The application never runs schema migrations automatically.

BEGIN;

CREATE TABLE public.password_reset_request (
    id bigserial PRIMARY KEY,
    user_id bigint NULL
        REFERENCES public.app_user(id) ON DELETE CASCADE,
    request_token_hash character varying(64) NOT NULL,
    email_hash character varying(64) NOT NULL,
    requester_ip_hash character varying(64) NOT NULL,
    code_hash character varying(64) NOT NULL,
    failed_attempts integer DEFAULT 0 NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    verified_at timestamp with time zone,
    reset_token_hash character varying(64),
    reset_token_expires_at timestamp with time zone,
    used_at timestamp with time zone,
    invalidated_at timestamp with time zone,
    delivery_failed_at timestamp with time zone,
    CONSTRAINT chk_password_reset_attempts
        CHECK (failed_attempts >= 0 AND failed_attempts <= 5),
    CONSTRAINT chk_password_reset_dates CHECK (expires_at > created_at),
    CONSTRAINT chk_password_reset_token_fields CHECK (
        (reset_token_hash IS NULL AND reset_token_expires_at IS NULL)
        OR
        (reset_token_hash IS NOT NULL AND reset_token_expires_at IS NOT NULL)
    ),
    CONSTRAINT password_reset_request_token_hash_key
        UNIQUE (request_token_hash),
    CONSTRAINT password_reset_reset_token_hash_key
        UNIQUE (reset_token_hash)
);

CREATE UNIQUE INDEX ux_password_reset_active_email
    ON public.password_reset_request (email_hash)
    WHERE used_at IS NULL AND invalidated_at IS NULL;

CREATE INDEX ix_password_reset_email_created
    ON public.password_reset_request (email_hash, created_at);

CREATE INDEX ix_password_reset_ip_created
    ON public.password_reset_request (requester_ip_hash, created_at);

CREATE INDEX ix_password_reset_user_id
    ON public.password_reset_request (user_id);

CREATE INDEX ix_password_reset_expires_at
    ON public.password_reset_request (expires_at);

COMMIT;
