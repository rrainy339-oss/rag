from __future__ import annotations

import unittest

from app.domain.permissions import (
    access_classification_level,
    access_principal_ids,
    access_visible_to_context,
    security_context_for_subject,
)
from app.domain.schemas import AccessControl


class PermissionTest(unittest.TestCase):
    def test_access_payload_principals_are_scoped_by_tenant(self) -> None:
        access = AccessControl(tenant_id="tenant-a", allowed_group_ids=["hr"])

        self.assertEqual(access_principal_ids(access), ["group:tenant-a:hr"])

    def test_public_classification_is_lowest_level(self) -> None:
        access = AccessControl(classification="public")

        self.assertEqual(access_classification_level(access), 0)

    def test_unscoped_group_acl_matches_tenanted_subject_group(self) -> None:
        access = AccessControl(allowed_group_ids=["hr"])
        context = security_context_for_subject(tenant_id="tenant-a", group_ids=["hr"])

        self.assertTrue(access_visible_to_context(access, context))

    def test_visibility_uses_principal_intersection_and_clearance(self) -> None:
        access = AccessControl(
            tenant_id="tenant-a",
            allowed_group_ids=["hr"],
            classification="confidential",
        )
        allowed = security_context_for_subject(
            tenant_id="tenant-a",
            group_ids=["hr"],
            max_classification="confidential",
        )
        wrong_group = security_context_for_subject(
            tenant_id="tenant-a",
            group_ids=["engineering"],
            max_classification="confidential",
        )
        low_clearance = security_context_for_subject(
            tenant_id="tenant-a",
            group_ids=["hr"],
            max_classification="internal",
        )

        self.assertTrue(access_visible_to_context(access, allowed))
        self.assertFalse(access_visible_to_context(access, wrong_group))
        self.assertFalse(access_visible_to_context(access, low_clearance))


if __name__ == "__main__":
    unittest.main()
