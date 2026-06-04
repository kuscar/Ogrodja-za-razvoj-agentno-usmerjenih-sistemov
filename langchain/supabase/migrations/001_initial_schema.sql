-- =====================================================================
-- 001_initial_schema.sql
-- Tables for the CV / Cover Letter Builder.
-- =====================================================================

create extension if not exists "uuid-ossp" schema extensions;
create extension if not exists vector schema extensions;       

create table if not exists public.profiles (
    user_id           uuid primary key references auth.users(id) on delete cascade,
    full_name         text not null,
    email             text not null,
    phone             text,
    location          text,
    hard_skills       text[] not null default '{}',
    soft_skills       text[] not null default '{}',
    certifications    text[] not null default '{}',
    leadership        text[] not null default '{}',
    extracurricular_activities text[] not null default '{}',
    created_at        timestamptz not null default now(),
    updated_at        timestamptz not null default now()
);

create table if not exists public.experiences (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references public.profiles(user_id) on delete cascade,
    company     text not null,
    title       text not null,
    start_date  text,
    end_date    text,
    bullets     text[] not null default '{}',
    created_at  timestamptz not null default now()
);

create index if not exists experiences_user_id_idx on public.experiences(user_id);

create table if not exists public.education (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references public.profiles(user_id) on delete cascade,
    institution text not null,
    degree      text not null,
    field       text,
    start_date  text,
    end_date    text
);

create index if not exists education_user_id_idx on public.education(user_id);

create table if not exists public.experience_embeddings (
    id          uuid primary key default gen_random_uuid(),
    user_id     uuid not null references public.profiles(user_id) on delete cascade,
    exp_id      uuid references public.experiences(id) on delete cascade,
    chunk       text not null,
    embedding   extensions.vector(768) not null,             
    created_at  timestamptz not null default now()
);

create index if not exists experience_embeddings_user_id_idx
    on public.experience_embeddings(user_id);

create index if not exists experience_embeddings_ivf_idx
    on public.experience_embeddings
    using ivfflat (embedding extensions.vector_cosine_ops)   
    with (lists = 100);


create or replace function public.match_experiences(
    p_user_id   uuid,
    p_query     extensions.vector(768),                        
    p_k         int default 8
) returns table (
    chunk        text,
    similarity   float,
    exp_id       uuid
)
language sql
stable
security invoker        
set search_path = public, extensions 
as $$
    select chunk,
           1 - (embedding <=> p_query) as similarity,
           exp_id
    from public.experience_embeddings
    where user_id = p_user_id       
    order by embedding <=> p_query
    limit p_k;
$$;