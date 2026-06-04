-- =====================================================================
-- 002_rls_policies.sql
-- =====================================================================

alter table public.profiles               enable row level security;
alter table public.experiences            enable row level security;
alter table public.education              enable row level security;
alter table public.experience_embeddings  enable row level security;
alter table public.profiles               force row level security;
alter table public.experiences            force row level security;
alter table public.education              force row level security;
alter table public.experience_embeddings  force row level security;

create policy profiles_select_own
    on public.profiles for select
    using (auth.uid() = user_id);

create policy profiles_modify_own
    on public.profiles for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy experiences_select_own
    on public.experiences for select
    using (auth.uid() = user_id);

create policy experiences_modify_own
    on public.experiences for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy education_select_own
    on public.education for select
    using (auth.uid() = user_id);

create policy education_modify_own
    on public.education for all
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

create policy embeddings_select_own
    on public.experience_embeddings for select
    using (auth.uid() = user_id);

create policy embeddings_insert_own
    on public.experience_embeddings for insert
    with check (auth.uid() = user_id);

create policy embeddings_delete_own
    on public.experience_embeddings for delete
    using (auth.uid() = user_id);

revoke all on public.experience_embeddings from anon, authenticated;
grant  select, insert, delete on public.experience_embeddings to authenticated;
