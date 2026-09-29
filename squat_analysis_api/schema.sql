--
-- PostgreSQL database dump
--

\restrict N2gsUieSvDhuPmU2OIrrrZHka4BsONYolpn03QHxtLtYTjMktI8DHjOYHzWZtOE

-- Dumped from database version 16.15 (Ubuntu 16.15-0ubuntu0.24.04.1)
-- Dumped by pg_dump version 16.15 (Ubuntu 16.15-0ubuntu0.24.04.1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: achievement; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.achievement (
    id bigint NOT NULL,
    code character varying(60) NOT NULL,
    name character varying(120) NOT NULL,
    description text,
    xp_reward integer DEFAULT 0 NOT NULL,
    criteria_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_achievement_xp CHECK ((xp_reward >= 0))
);


--
-- Name: achievement_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.achievement_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: achievement_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.achievement_id_seq OWNED BY public.achievement.id;


--
-- Name: app_user; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.app_user (
    id bigint NOT NULL,
    email character varying(255) NOT NULL,
    password_hash text NOT NULL,
    full_name character varying(150),
    birth_date date,
    height_cm numeric(5,2),
    weight_kg numeric(5,2),
    training_level character varying(20),
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_app_user_height CHECK (((height_cm IS NULL) OR (height_cm > (0)::numeric))),
    CONSTRAINT chk_app_user_training_level CHECK (((training_level IS NULL) OR ((training_level)::text = ANY ((ARRAY['beginner'::character varying, 'intermediate'::character varying, 'advanced'::character varying])::text[])))),
    CONSTRAINT chk_app_user_weight CHECK (((weight_kg IS NULL) OR (weight_kg > (0)::numeric)))
);


--
-- Name: app_user_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.app_user_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: app_user_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.app_user_id_seq OWNED BY public.app_user.id;


--
-- Name: auth_session; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.auth_session (
    id bigint NOT NULL,
    user_id bigint NOT NULL,
    token_hash text NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    expires_at timestamp with time zone NOT NULL,
    last_used_at timestamp with time zone,
    revoked_at timestamp with time zone,
    CONSTRAINT chk_auth_session_dates CHECK ((expires_at > created_at))
);


--
-- Name: auth_session_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.auth_session_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: auth_session_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.auth_session_id_seq OWNED BY public.auth_session.id;


--
-- Name: challenge; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.challenge (
    id bigint NOT NULL,
    code character varying(60) NOT NULL,
    title character varying(120) NOT NULL,
    description text,
    target_type character varying(50) NOT NULL,
    target_value integer NOT NULL,
    xp_reward integer DEFAULT 0 NOT NULL,
    criteria_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    start_at timestamp with time zone,
    end_at timestamp with time zone,
    is_active boolean DEFAULT true NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_challenge_dates CHECK (((start_at IS NULL) OR (end_at IS NULL) OR (end_at > start_at))),
    CONSTRAINT chk_challenge_target CHECK ((target_value > 0)),
    CONSTRAINT chk_challenge_xp CHECK ((xp_reward >= 0))
);


--
-- Name: challenge_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.challenge_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: challenge_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.challenge_id_seq OWNED BY public.challenge.id;


--
-- Name: evaluation; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.evaluation (
    id bigint NOT NULL,
    user_id bigint NOT NULL,
    job_id character varying(64) NOT NULL,
    exercise character varying(30) DEFAULT 'squat'::character varying NOT NULL,
    mode character varying(20) DEFAULT 'beginner'::character varying NOT NULL,
    status character varying(20) DEFAULT 'queued'::character varying NOT NULL,
    input_filename text,
    correct boolean,
    total_reps integer,
    correct_reps integer,
    incorrect_reps integer,
    original_video_url text,
    annotated_video_url text,
    video_seconds numeric(10,2),
    assistant_seconds numeric(10,2),
    error text,
    report_json jsonb DEFAULT '{}'::jsonb NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    CONSTRAINT chk_evaluation_correct_reps CHECK (((correct_reps IS NULL) OR (correct_reps >= 0))),
    CONSTRAINT chk_evaluation_exercise CHECK (((exercise)::text = 'squat'::text)),
    CONSTRAINT chk_evaluation_incorrect_reps CHECK (((incorrect_reps IS NULL) OR (incorrect_reps >= 0))),
    CONSTRAINT chk_evaluation_mode CHECK (((mode)::text = ANY ((ARRAY['beginner'::character varying, 'pro'::character varying])::text[]))),
    CONSTRAINT chk_evaluation_rep_totals CHECK (((total_reps IS NULL) OR (correct_reps IS NULL) OR (incorrect_reps IS NULL) OR (total_reps = (correct_reps + incorrect_reps)))),
    CONSTRAINT chk_evaluation_status CHECK (((status)::text = ANY ((ARRAY['queued'::character varying, 'processing'::character varying, 'completed'::character varying, 'failed'::character varying])::text[]))),
    CONSTRAINT chk_evaluation_total_reps CHECK (((total_reps IS NULL) OR (total_reps >= 0)))
);


--
-- Name: evaluation_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.evaluation_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: evaluation_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.evaluation_id_seq OWNED BY public.evaluation.id;


--
-- Name: user_achievement; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_achievement (
    id bigint NOT NULL,
    user_id bigint NOT NULL,
    achievement_id bigint NOT NULL,
    evaluation_id bigint,
    unlocked_at timestamp with time zone DEFAULT now() NOT NULL
);


--
-- Name: user_achievement_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.user_achievement_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: user_achievement_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.user_achievement_id_seq OWNED BY public.user_achievement.id;


--
-- Name: user_challenge; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_challenge (
    id bigint NOT NULL,
    user_id bigint NOT NULL,
    challenge_id bigint NOT NULL,
    progress_value integer DEFAULT 0 NOT NULL,
    status character varying(20) DEFAULT 'active'::character varying NOT NULL,
    started_at timestamp with time zone DEFAULT now() NOT NULL,
    completed_at timestamp with time zone,
    claimed_at timestamp with time zone,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_user_challenge_progress CHECK ((progress_value >= 0)),
    CONSTRAINT chk_user_challenge_status CHECK (((status)::text = ANY ((ARRAY['active'::character varying, 'completed'::character varying, 'claimed'::character varying, 'expired'::character varying])::text[])))
);


--
-- Name: user_challenge_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.user_challenge_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: user_challenge_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.user_challenge_id_seq OWNED BY public.user_challenge.id;


--
-- Name: user_progress; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.user_progress (
    user_id bigint NOT NULL,
    total_xp integer DEFAULT 0 NOT NULL,
    level integer DEFAULT 1 NOT NULL,
    current_streak integer DEFAULT 0 NOT NULL,
    longest_streak integer DEFAULT 0 NOT NULL,
    total_evaluations integer DEFAULT 0 NOT NULL,
    total_correct_reps integer DEFAULT 0 NOT NULL,
    total_incorrect_reps integer DEFAULT 0 NOT NULL,
    updated_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_user_progress_counts CHECK (((total_evaluations >= 0) AND (total_correct_reps >= 0) AND (total_incorrect_reps >= 0))),
    CONSTRAINT chk_user_progress_level CHECK ((level >= 1)),
    CONSTRAINT chk_user_progress_streak CHECK (((current_streak >= 0) AND (longest_streak >= 0))),
    CONSTRAINT chk_user_progress_xp CHECK ((total_xp >= 0))
);


--
-- Name: xp_transaction; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.xp_transaction (
    id bigint NOT NULL,
    user_id bigint NOT NULL,
    evaluation_id bigint,
    amount integer NOT NULL,
    reason character varying(100) NOT NULL,
    created_at timestamp with time zone DEFAULT now() NOT NULL,
    CONSTRAINT chk_xp_transaction_amount CHECK ((amount <> 0))
);


--
-- Name: xp_transaction_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE public.xp_transaction_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: xp_transaction_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE public.xp_transaction_id_seq OWNED BY public.xp_transaction.id;


--
-- Name: achievement id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.achievement ALTER COLUMN id SET DEFAULT nextval('public.achievement_id_seq'::regclass);


--
-- Name: app_user id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.app_user ALTER COLUMN id SET DEFAULT nextval('public.app_user_id_seq'::regclass);


--
-- Name: auth_session id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.auth_session ALTER COLUMN id SET DEFAULT nextval('public.auth_session_id_seq'::regclass);


--
-- Name: challenge id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenge ALTER COLUMN id SET DEFAULT nextval('public.challenge_id_seq'::regclass);


--
-- Name: evaluation id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation ALTER COLUMN id SET DEFAULT nextval('public.evaluation_id_seq'::regclass);


--
-- Name: user_achievement id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_achievement ALTER COLUMN id SET DEFAULT nextval('public.user_achievement_id_seq'::regclass);


--
-- Name: user_challenge id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_challenge ALTER COLUMN id SET DEFAULT nextval('public.user_challenge_id_seq'::regclass);


--
-- Name: xp_transaction id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.xp_transaction ALTER COLUMN id SET DEFAULT nextval('public.xp_transaction_id_seq'::regclass);


--
-- Name: achievement achievement_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.achievement
    ADD CONSTRAINT achievement_code_key UNIQUE (code);


--
-- Name: achievement achievement_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.achievement
    ADD CONSTRAINT achievement_pkey PRIMARY KEY (id);


--
-- Name: app_user app_user_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.app_user
    ADD CONSTRAINT app_user_pkey PRIMARY KEY (id);


--
-- Name: auth_session auth_session_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.auth_session
    ADD CONSTRAINT auth_session_pkey PRIMARY KEY (id);


--
-- Name: auth_session auth_session_token_hash_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.auth_session
    ADD CONSTRAINT auth_session_token_hash_key UNIQUE (token_hash);


--
-- Name: challenge challenge_code_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenge
    ADD CONSTRAINT challenge_code_key UNIQUE (code);


--
-- Name: challenge challenge_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.challenge
    ADD CONSTRAINT challenge_pkey PRIMARY KEY (id);


--
-- Name: evaluation evaluation_job_id_key; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation
    ADD CONSTRAINT evaluation_job_id_key UNIQUE (job_id);


--
-- Name: evaluation evaluation_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation
    ADD CONSTRAINT evaluation_pkey PRIMARY KEY (id);


--
-- Name: user_achievement user_achievement_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_achievement
    ADD CONSTRAINT user_achievement_pkey PRIMARY KEY (id);


--
-- Name: user_challenge user_challenge_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_challenge
    ADD CONSTRAINT user_challenge_pkey PRIMARY KEY (id);


--
-- Name: user_progress user_progress_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_progress
    ADD CONSTRAINT user_progress_pkey PRIMARY KEY (user_id);


--
-- Name: user_achievement ux_user_achievement; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_achievement
    ADD CONSTRAINT ux_user_achievement UNIQUE (user_id, achievement_id);


--
-- Name: user_challenge ux_user_challenge; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_challenge
    ADD CONSTRAINT ux_user_challenge UNIQUE (user_id, challenge_id);


--
-- Name: xp_transaction xp_transaction_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.xp_transaction
    ADD CONSTRAINT xp_transaction_pkey PRIMARY KEY (id);


--
-- Name: ix_auth_session_expires_at; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_auth_session_expires_at ON public.auth_session USING btree (expires_at);


--
-- Name: ix_auth_session_user_id; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_auth_session_user_id ON public.auth_session USING btree (user_id);


--
-- Name: ix_evaluation_report_json; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_evaluation_report_json ON public.evaluation USING gin (report_json);


--
-- Name: ix_evaluation_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_evaluation_status ON public.evaluation USING btree (status);


--
-- Name: ix_evaluation_user_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_evaluation_user_created ON public.evaluation USING btree (user_id, created_at DESC);


--
-- Name: ix_user_achievement_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_user_achievement_user ON public.user_achievement USING btree (user_id);


--
-- Name: ix_user_challenge_status; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_user_challenge_status ON public.user_challenge USING btree (status);


--
-- Name: ix_user_challenge_user; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_user_challenge_user ON public.user_challenge USING btree (user_id);


--
-- Name: ix_xp_transaction_evaluation; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_xp_transaction_evaluation ON public.xp_transaction USING btree (evaluation_id);


--
-- Name: ix_xp_transaction_user_created; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX ix_xp_transaction_user_created ON public.xp_transaction USING btree (user_id, created_at DESC);


--
-- Name: ux_app_user_email; Type: INDEX; Schema: public; Owner: -
--

CREATE UNIQUE INDEX ux_app_user_email ON public.app_user USING btree (lower((email)::text));


--
-- Name: auth_session auth_session_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.auth_session
    ADD CONSTRAINT auth_session_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.app_user(id) ON DELETE CASCADE;


--
-- Name: evaluation evaluation_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.evaluation
    ADD CONSTRAINT evaluation_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.app_user(id) ON DELETE CASCADE;


--
-- Name: user_achievement user_achievement_achievement_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_achievement
    ADD CONSTRAINT user_achievement_achievement_id_fkey FOREIGN KEY (achievement_id) REFERENCES public.achievement(id) ON DELETE CASCADE;


--
-- Name: user_achievement user_achievement_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_achievement
    ADD CONSTRAINT user_achievement_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES public.evaluation(id) ON DELETE SET NULL;


--
-- Name: user_achievement user_achievement_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_achievement
    ADD CONSTRAINT user_achievement_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.app_user(id) ON DELETE CASCADE;


--
-- Name: user_challenge user_challenge_challenge_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_challenge
    ADD CONSTRAINT user_challenge_challenge_id_fkey FOREIGN KEY (challenge_id) REFERENCES public.challenge(id) ON DELETE CASCADE;


--
-- Name: user_challenge user_challenge_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_challenge
    ADD CONSTRAINT user_challenge_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.app_user(id) ON DELETE CASCADE;


--
-- Name: user_progress user_progress_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.user_progress
    ADD CONSTRAINT user_progress_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.app_user(id) ON DELETE CASCADE;


--
-- Name: xp_transaction xp_transaction_evaluation_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.xp_transaction
    ADD CONSTRAINT xp_transaction_evaluation_id_fkey FOREIGN KEY (evaluation_id) REFERENCES public.evaluation(id) ON DELETE SET NULL;


--
-- Name: xp_transaction xp_transaction_user_id_fkey; Type: FK CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY public.xp_transaction
    ADD CONSTRAINT xp_transaction_user_id_fkey FOREIGN KEY (user_id) REFERENCES public.app_user(id) ON DELETE CASCADE;


--
-- PostgreSQL database dump complete
--

\unrestrict N2gsUieSvDhuPmU2OIrrrZHka4BsONYolpn03QHxtLtYTjMktI8DHjOYHzWZtOE

