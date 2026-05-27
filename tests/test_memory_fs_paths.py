"""Unit tests for blipshell.memory.fs_paths."""

import pytest

from blipshell.memory.fs_paths import (
    MAX_PATH_DEPTH,
    MemoryPath,
    PathError,
    Tier,
    build_filename,
    check_permission,
    parse,
    requires_approval,
    slugify,
)


class TestSlugify:
    def test_basic(self):
        assert slugify("Hello World") == "hello-world"

    def test_special_chars(self):
        assert slugify("Don't quaff unknown potions!") == "don-t-quaff-unknown-potions"

    def test_collapse_runs(self):
        assert slugify("a  b   c") == "a-b-c"

    def test_strip_edges(self):
        assert slugify("---a---") == "a"

    def test_empty(self):
        assert slugify("") == "untitled"
        assert slugify("!!!") == "untitled"

    def test_length_cap(self):
        text = "word-" * 30  # well past 60 chars
        result = slugify(text, max_length=20)
        assert len(result) <= 20
        # Should cut at hyphen boundary if possible.
        assert not result.endswith("-")

    def test_unicode(self):
        # Non-ASCII gets stripped to safe chars.
        assert slugify("café résumé") == "caf-r-sum"


class TestBuildFilename:
    def test_combines_id_and_slug(self):
        assert build_filename(142, "Never skip tests") == "142-never-skip-tests.md"

    def test_handles_empty_text(self):
        assert build_filename(7, "") == "7-untitled.md"


class TestParseRoot:
    def test_memories_no_slash(self):
        p = parse("/memories")
        assert p.is_root
        assert p.tier is None

    def test_memories_with_slash(self):
        p = parse("/memories/")
        assert p.is_root


class TestParseTierListing:
    @pytest.mark.parametrize("tier", [t.value for t in Tier])
    def test_each_tier(self, tier):
        p = parse(f"/memories/{tier}")
        assert p.tier == Tier(tier)
        assert p.is_directory
        assert p.filename is None

    def test_trailing_slash(self):
        p = parse("/memories/lessons/")
        assert p.tier == Tier.LESSONS
        assert p.is_directory


class TestParseLessons:
    def test_project_listing(self):
        p = parse("/memories/lessons/blipshell")
        assert p.tier == Tier.LESSONS
        assert p.project == "blipshell"
        assert p.is_directory

    def test_project_listing_trailing_slash(self):
        p = parse("/memories/lessons/blipshell/")
        assert p.tier == Tier.LESSONS
        assert p.project == "blipshell"

    def test_full_file_path(self):
        p = parse("/memories/lessons/blipshell/142-never-skip-tests.md")
        assert p.tier == Tier.LESSONS
        assert p.project == "blipshell"
        assert p.filename == "142-never-skip-tests.md"
        assert p.file_id == 142
        assert p.slug == "never-skip-tests"

    def test_filename_without_id(self):
        p = parse("/memories/lessons/blipshell/never-skip-tests.md")
        assert p.file_id is None
        assert p.slug == "never-skip-tests"

    def test_invalid_project_name(self):
        # Generic segment validator catches uppercase/spaces before tier-specific check.
        with pytest.raises(PathError, match="Invalid path segment|Invalid project"):
            parse("/memories/lessons/Bad Project Name/x.md")

    def test_too_deep(self):
        with pytest.raises(PathError, match="too deep|Invalid lessons"):
            parse("/memories/lessons/blipshell/subdir/142-foo.md")


class TestParseCore:
    def test_file(self):
        p = parse("/memories/core/7-user-prefers-terse.md")
        assert p.tier == Tier.CORE
        assert p.file_id == 7
        assert p.slug == "user-prefers-terse"

    def test_rejects_subdir(self):
        with pytest.raises(PathError, match="does not support subdirectories"):
            parse("/memories/core/personality/x.md")


class TestParseDigests:
    def test_file(self):
        p = parse("/memories/digests/blipshell.md")
        assert p.tier == Tier.DIGESTS
        assert p.project == "blipshell"
        assert p.filename == "blipshell.md"

    def test_rejects_no_extension(self):
        with pytest.raises(PathError, match="must end with .md"):
            parse("/memories/digests/blipshell")


class TestParseNotes:
    def test_tier_listing(self):
        p = parse("/memories/notes/")
        assert p.tier == Tier.NOTES
        assert p.is_directory

    def test_file(self):
        p = parse("/memories/notes/plan.md")
        assert p.tier == Tier.NOTES
        assert p.filename == "plan.md"
        assert p.slug == "plan"

    def test_rejects_subdir(self):
        with pytest.raises(PathError, match="flat"):
            parse("/memories/notes/123/plan.md")

    def test_rejects_no_extension(self):
        with pytest.raises(PathError, match="must end with .md"):
            parse("/memories/notes/plan")

    def test_rejects_bad_name(self):
        with pytest.raises(PathError, match="Invalid note name"):
            parse("/memories/notes/has spaces.md")


class TestParseRejects:
    def test_empty(self):
        with pytest.raises(PathError, match="empty"):
            parse("")

    def test_none(self):
        with pytest.raises(PathError, match="required"):
            parse(None)

    def test_traversal(self):
        with pytest.raises(PathError, match="traversal"):
            parse("/memories/lessons/../etc/passwd")

    def test_backslash(self):
        with pytest.raises(PathError, match="forward slashes"):
            parse("/memories\\lessons\\x.md")

    def test_null_byte(self):
        with pytest.raises(PathError, match="Null bytes"):
            parse("/memories/lessons\x00/x.md")

    def test_no_leading_slash(self):
        with pytest.raises(PathError, match="must start with /memories"):
            parse("memories/lessons")

    def test_wrong_prefix(self):
        with pytest.raises(PathError, match="must start with /memories"):
            parse("/etc/passwd")

    def test_unknown_tier(self):
        with pytest.raises(PathError, match="Unknown tier"):
            parse("/memories/notatier/x.md")

    def test_too_deep_segments(self):
        path = "/memories/" + "/".join("a" for _ in range(MAX_PATH_DEPTH + 1))
        with pytest.raises(PathError, match="too deep"):
            parse(path)


class TestPermissions:
    def test_lessons_read_only(self):
        assert check_permission(Tier.LESSONS, "read")
        for op in ("create", "edit", "delete"):
            assert not check_permission(Tier.LESSONS, op)

    def test_core_writes_require_approval(self):
        assert check_permission(Tier.CORE, "read")
        assert not requires_approval(Tier.CORE, "read")
        for op in ("create", "edit", "delete"):
            assert check_permission(Tier.CORE, op)
            assert requires_approval(Tier.CORE, op)

    def test_digests_read_only(self):
        assert check_permission(Tier.DIGESTS, "read")
        assert not check_permission(Tier.DIGESTS, "create")
        assert not check_permission(Tier.DIGESTS, "edit")
        assert not check_permission(Tier.DIGESTS, "delete")

    def test_sessions_read_only(self):
        assert check_permission(Tier.SESSIONS, "read")
        assert not check_permission(Tier.SESSIONS, "edit")

    def test_friction_read_only(self):
        assert check_permission(Tier.FRICTION, "read")
        assert not check_permission(Tier.FRICTION, "edit")

    def test_notes_full_no_approval(self):
        for op in ("read", "create", "edit", "delete"):
            assert check_permission(Tier.NOTES, op)
            assert not requires_approval(Tier.NOTES, op)
