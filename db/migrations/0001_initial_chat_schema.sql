-- Runtime queries must use a dedicated non-owner role without BYPASSRLS.
-- Set tara.user_id transaction-locally only after authenticating the request.
BEGIN;
CREATE TABLE IF NOT EXISTS app_users(id uuid PRIMARY KEY,display_name text NOT NULL DEFAULT '',created_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS conversations(id uuid PRIMARY KEY DEFAULT gen_random_uuid(),owner_id uuid NOT NULL REFERENCES app_users(id) ON DELETE CASCADE,title text NOT NULL DEFAULT 'New conversation',created_at timestamptz NOT NULL DEFAULT now(),updated_at timestamptz NOT NULL DEFAULT now());
CREATE TABLE IF NOT EXISTS messages(id uuid PRIMARY KEY DEFAULT gen_random_uuid(),conversation_id uuid NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,role text NOT NULL CHECK(role IN ('user','assistant','system')),content text NOT NULL,created_at timestamptz NOT NULL DEFAULT now());
CREATE INDEX IF NOT EXISTS conversations_owner_updated_idx ON conversations(owner_id,updated_at DESC);
CREATE INDEX IF NOT EXISTS messages_conversation_created_idx ON messages(conversation_id,created_at ASC);
ALTER TABLE app_users ENABLE ROW LEVEL SECURITY; ALTER TABLE app_users FORCE ROW LEVEL SECURITY;
ALTER TABLE conversations ENABLE ROW LEVEL SECURITY; ALTER TABLE conversations FORCE ROW LEVEL SECURITY;
ALTER TABLE messages ENABLE ROW LEVEL SECURITY; ALTER TABLE messages FORCE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS app_users_select_self ON app_users; CREATE POLICY app_users_select_self ON app_users FOR SELECT USING(id=NULLIF(current_setting('tara.user_id',true),'')::uuid);
DROP POLICY IF EXISTS app_users_insert_self ON app_users; CREATE POLICY app_users_insert_self ON app_users FOR INSERT WITH CHECK(id=NULLIF(current_setting('tara.user_id',true),'')::uuid);
DROP POLICY IF EXISTS app_users_update_self ON app_users; CREATE POLICY app_users_update_self ON app_users FOR UPDATE USING(id=NULLIF(current_setting('tara.user_id',true),'')::uuid) WITH CHECK(id=NULLIF(current_setting('tara.user_id',true),'')::uuid);
DROP POLICY IF EXISTS conversations_select_owner ON conversations; CREATE POLICY conversations_select_owner ON conversations FOR SELECT USING(owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid);
DROP POLICY IF EXISTS conversations_insert_owner ON conversations; CREATE POLICY conversations_insert_owner ON conversations FOR INSERT WITH CHECK(owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid);
DROP POLICY IF EXISTS conversations_update_owner ON conversations; CREATE POLICY conversations_update_owner ON conversations FOR UPDATE USING(owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid) WITH CHECK(owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid);
DROP POLICY IF EXISTS conversations_delete_owner ON conversations; CREATE POLICY conversations_delete_owner ON conversations FOR DELETE USING(owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid);
DROP POLICY IF EXISTS messages_select_owner ON messages; CREATE POLICY messages_select_owner ON messages FOR SELECT USING(EXISTS(SELECT 1 FROM conversations c WHERE c.id=messages.conversation_id AND c.owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid));
DROP POLICY IF EXISTS messages_insert_owner ON messages; CREATE POLICY messages_insert_owner ON messages FOR INSERT WITH CHECK(EXISTS(SELECT 1 FROM conversations c WHERE c.id=messages.conversation_id AND c.owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid));
DROP POLICY IF EXISTS messages_update_owner ON messages; CREATE POLICY messages_update_owner ON messages FOR UPDATE USING(EXISTS(SELECT 1 FROM conversations c WHERE c.id=messages.conversation_id AND c.owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid)) WITH CHECK(EXISTS(SELECT 1 FROM conversations c WHERE c.id=messages.conversation_id AND c.owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid));
DROP POLICY IF EXISTS messages_delete_owner ON messages; CREATE POLICY messages_delete_owner ON messages FOR DELETE USING(EXISTS(SELECT 1 FROM conversations c WHERE c.id=messages.conversation_id AND c.owner_id=NULLIF(current_setting('tara.user_id',true),'')::uuid));
COMMIT;
