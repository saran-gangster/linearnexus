#!/bin/bash
# Automatic git commit script
# Usage: ./commit.sh "your commit message"
#        ./commit.sh (will use default message with timestamp)

set -e

# Check if git repo
if [ ! -d .git ]; then
    echo "Error: Not a git repository"
    exit 1
fi

# Get commit message from argument or use default
if [ -z "$1" ]; then
    COMMIT_MSG="Auto-commit: $(date '+%Y-%m-%d %H:%M:%S')"
else
    COMMIT_MSG="$1"
fi

# Show status
echo "=== Git Status ==="
git status --short

# Add all changes
echo ""
echo "=== Adding all changes ==="
git add .

# Show what will be committed
echo ""
echo "=== Changes to be committed ==="
git status --short

# Commit
echo ""
echo "=== Committing with message: $COMMIT_MSG ==="
git commit -m "$COMMIT_MSG"

# Show last commit
echo ""
echo "=== Last Commit ==="
git log -1 --oneline

echo ""
echo "✓ Commit successful!"
echo ""
echo "To push to remote, run: git push"
