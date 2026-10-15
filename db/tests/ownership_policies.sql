-- Run against a disposable migrated database using a role allowed to create roles.
-- This transaction rolls back the temporary role and all fixtures.
\set ON_ERROR_STOP on
BEGIN;
CREATE ROLE tara_app_test NOLOGIN;
GRANT SELECT,INSERT,UPDATE,DELETE ON app_users,conversations,messages TO tara_app_test;
INSERT INTO app_users(id,display_name) VALUES('10000000-0000-4000-8000-000000000001','Owner A'),('10000000-0000-4000-8000-000000000002','Owner B');
INSERT INTO conversations(id,owner_id,title) VALUES('20000000-0000-4000-8000-000000000001','10000000-0000-4000-8000-000000000001','A private chat'),('20000000-0000-4000-8000-000000000002','10000000-0000-4000-8000-000000000002','B private chat');
INSERT INTO messages(id,conversation_id,role,content) VALUES('30000000-0000-4000-8000-000000000001','20000000-0000-4000-8000-000000000001','user','A secret'),('30000000-0000-4000-8000-000000000002','20000000-0000-4000-8000-000000000002','user','B secret');
SET LOCAL ROLE tara_app_test; SELECT set_config('tara.user_id','10000000-0000-4000-8000-000000000001',true);
DO $test$ DECLARE n integer; BEGIN
SELECT count(*) INTO n FROM app_users; IF n<>1 THEN RAISE EXCEPTION 'user visibility expected 1, got %',n; END IF;
SELECT count(*) INTO n FROM conversations; IF n<>1 THEN RAISE EXCEPTION 'conversation visibility expected 1, got %',n; END IF;
SELECT count(*) INTO n FROM messages; IF n<>1 THEN RAISE EXCEPTION 'message visibility expected 1, got %',n; END IF;
INSERT INTO messages(conversation_id,role,content) VALUES('20000000-0000-4000-8000-000000000001','assistant','Allowed');
BEGIN INSERT INTO messages(conversation_id,role,content) VALUES('20000000-0000-4000-8000-000000000002','assistant','Blocked'); RAISE EXCEPTION 'cross-owner insert unexpectedly succeeded'; EXCEPTION WHEN insufficient_privilege THEN NULL; END;
UPDATE conversations SET title='Should not update' WHERE id='20000000-0000-4000-8000-000000000002'; GET DIAGNOSTICS n=ROW_COUNT; IF n<>0 THEN RAISE EXCEPTION 'cross-owner update changed % rows',n; END IF;
END $test$; ROLLBACK;
