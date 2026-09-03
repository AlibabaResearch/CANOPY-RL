#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@desc swe-agent 相关常量配置
@author: plm
@create: 2026-04-03
"""

RESET_GIT_LOG_COMMAND = r"""
#!/bin/bash

# Default value
REMOVE_TAG=true

# Parse parameters
while [[ $# -gt 0 ]]; do
    case $1 in
        --remove_tag)
            if [ "$2" = "true" ] || [ "$2" = "false" ]; then
                REMOVE_TAG=$2
                shift 2
            else
                echo "Error: --remove_tag parameter must be true or false"
                exit 1
            fi
            ;;
        *)
            TARGET_COMMIT=$1
            shift
            ;;
    esac
done

# Check if commit_id is provided
if [ -z "$TARGET_COMMIT" ]; then
    echo "Usage: $0 <commit_id> [--remove_tag true|false]"
    echo "  --remove_tag: Whether to delete tags (Default: true)"
    exit 1
fi

echo "--- 1. Protecting local modifications (Stashing) ---"
# --include-untracked will stash untracked files together
# If there are no modifications, stash will return failure, so add a check
STASH_RESULT=$(git stash push --include-untracked -m "Pre-sanitization backup")

echo "--- 2. Starting to clean up future information ---"
# Force reset to the target commit
# git reset --hard $TARGET_COMMIT

# Remove remote repository and other branches
git remote remove origin 2>/dev/null
git branch | grep -v "^\*" | xargs -r git branch -D

# Determine whether to delete tags based on parameters
if [ "$REMOVE_TAG" = "true" ]; then
    echo "Deleting tags..."
    git tag | xargs -r git tag -d
else
    echo "Skipping tag deletion (--remove_tag=false)"
fi

# Completely clean up reflog and objects to prevent leakage
git reflog expire --expire=now --all
git gc --prune=now

echo "--- 3. Restoring local modifications (Unstashing) ---"
# If stash was successful before, restore it now
if [[ "$STASH_RESULT" != "No local changes to save" ]]; then
    git stash pop
    echo "Local modifications have been restored."
else
    echo "No modifications requiring preservation were detected earlier."
fi

echo "--- Cleanup completed! Locked to $TARGET_COMMIT ---"

"""


RESET_GIT_LOG_COMMAND_SIMPLE_FORCE = """
rm -rf .git
git init
git add .
git commit -m "Initial commit for agent task"
"""
