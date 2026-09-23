-- One-time Supabase setup for the Soft Sensor Toolbox.
-- Run in the Supabase dashboard: SQL Editor -> New query -> paste -> Run.
-- Safe to run again.

-- 1. Private bucket for dataset versions and model artifacts.
--    Only the backend (service-role key) reads or writes it.
insert into storage.buckets (id, name, public, file_size_limit)
values ('sst-data', 'sst-data', false, 104857600)
on conflict (id) do update set public = false, file_size_limit = excluded.file_size_limit;

-- 2. Row-level security with no policies on the app tables.
--    The backend also does this on every start (backend/database.py), but the
--    tables only exist after its first start; this covers a manual re-check.
do $$
declare t text;
begin
  foreach t in array array['datasets', 'dataset_versions', 'trained_models', 'project_state'] loop
    if to_regclass('public.' || t) is not null then
      execute format('alter table public.%I enable row level security', t);
    end if;
  end loop;
end $$;

-- 3. Check: every row should say rls_enabled = true and policies = 0.
select c.relname as table_name,
       c.relrowsecurity as rls_enabled,
       (select count(*) from pg_policies p where p.schemaname = 'public' and p.tablename = c.relname) as policies
from pg_class c
join pg_namespace n on n.oid = c.relnamespace
where n.nspname = 'public'
  and c.relname in ('datasets', 'dataset_versions', 'trained_models', 'project_state');
