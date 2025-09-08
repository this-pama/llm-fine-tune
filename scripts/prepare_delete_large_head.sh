#!/bin/bash
set -e

# prepare_delete_large_head.sh
# 
# Documentation script for safely removing large files from current HEAD.
# 
# ⚠️  IMPORTANT: This is a NON-DESTRUCTIVE operation that only affects the current branch.
# ⚠️  This does NOT rewrite git history or purge files from previous commits.
# ⚠️  Large files will still exist in git history and can be recovered.
#
# This script documents the exact steps to remove large files from the current
# working tree and prevent them from being re-committed in the future.

echo "🧹 Large File Cleanup Script (Non-Destructive)"
echo "=============================================="
echo ""
echo "⚠️  WARNING: This script will remove large files from the current HEAD only."
echo "⚠️  Git history is NOT modified. Files remain in previous commits."
echo "⚠️  This is a safe, non-destructive operation."
echo ""

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Error: Not in a git repository"
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet || ! git diff --staged --quiet; then
    echo "❌ Error: You have uncommitted changes. Please commit or stash them first."
    echo "Current status:"
    git status --porcelain
    exit 1
fi

echo "📋 Step 1: Identify large files currently tracked"
echo "-----------------------------------------------"

# Find files larger than 1MB that are currently tracked
echo "Searching for files larger than 1MB..."
large_files=$(git ls-files | xargs ls -l 2>/dev/null | awk '$5 > 1048576 {print $9, "(" $5 " bytes)"}' || true)

if [ -z "$large_files" ]; then
    echo "✅ No large files found in current HEAD"
else
    echo "📁 Large files found:"
    echo "$large_files"
fi

echo ""
echo "📋 Step 2: Identify data file patterns to ignore"
echo "-----------------------------------------------"

# Find CSV and JSONL files
data_files=$(find . -name "*.csv" -o -name "*.jsonl" | grep -v ".git" | grep -v "data/sample_sft_small.jsonl" || true)

if [ -z "$data_files" ]; then
    echo "✅ No data files found to remove"
else
    echo "📄 Data files found:"
    echo "$data_files"
fi

echo ""
echo "📋 Step 3: Actions that would be taken"
echo "-------------------------------------"

# Show what would be done without actually doing it
if [ -n "$large_files" ] || [ -n "$data_files" ]; then
    echo "The following commands would be executed:"
    echo ""
    echo "# Remove large files from git tracking (keeps local copies)"
    
    if [ -n "$large_files" ]; then
        echo "$large_files" | cut -d' ' -f1 | while read -r file; do
            if [ -f "$file" ]; then
                echo "git rm --cached '$file'"
            fi
        done
    fi
    
    if [ -n "$data_files" ]; then
        echo "$data_files" | while read -r file; do
            echo "git rm --cached '$file'"
        done
    fi
    
    echo ""
    echo "# Update .gitignore to prevent re-adding"
    echo "cat >> .gitignore << 'EOF'"
    echo "# Data files"
    echo "data/archive/*"
    echo "*.jsonl"
    echo "*.csv"
    echo "output_models/"
    echo "checkpoints/"
    echo "offload_folder/"
    echo "EOF"
    
    echo ""
    echo "# Commit the changes"
    echo "git add .gitignore"
    echo "git commit -m 'chore: remove large data files from HEAD and ignore'"
    
else
    echo "✅ No actions needed - no large files found"
fi

echo ""
echo "📋 Step 4: Execution (Interactive)"
echo "---------------------------------"

if [ -n "$large_files" ] || [ -n "$data_files" ]; then
    read -p "🤔 Do you want to execute these commands now? (y/N): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        echo ""
        echo "🏃 Executing cleanup commands..."
        
        # Remove large files from tracking
        if [ -n "$large_files" ]; then
            echo "$large_files" | cut -d' ' -f1 | while read -r file; do
                if [ -f "$file" ] && git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
                    echo "Removing from git: $file"
                    git rm --cached "$file"
                fi
            done
        fi
        
        if [ -n "$data_files" ]; then
            echo "$data_files" | while read -r file; do
                if git ls-files --error-unmatch "$file" >/dev/null 2>&1; then
                    echo "Removing from git: $file"
                    git rm --cached "$file"
                fi
            done
        fi
        
        # Update .gitignore if not already updated
        if ! grep -q "data/archive/\*" .gitignore 2>/dev/null; then
            echo ""
            echo "Updating .gitignore..."
            cat >> .gitignore << 'EOF'

# Data files (added by cleanup script)
data/archive/*
*.jsonl
*.csv
output_models/
checkpoints/
offload_folder/
EOF
        fi
        
        # Commit changes
        if git diff --staged --quiet; then
            echo "✅ No changes to commit"
        else
            echo ""
            echo "Committing changes..."
            git add .gitignore
            git commit -m "chore: remove large data files from HEAD and ignore"
            echo "✅ Changes committed successfully"
        fi
        
    else
        echo "❌ Cleanup cancelled by user"
    fi
else
    echo "✅ No cleanup needed"
fi

echo ""
echo "📋 Step 5: Verification"
echo "----------------------"

# Check final state
echo "Current repository size (working directory):"
du -sh . | grep -v ".git"

echo ""
echo "Files larger than 1MB still tracked:"
remaining_large=$(git ls-files | xargs ls -l 2>/dev/null | awk '$5 > 1048576 {print $9, "(" $5 " bytes)"}' || true)
if [ -z "$remaining_large" ]; then
    echo "✅ No large files remaining in git tracking"
else
    echo "⚠️  Large files still tracked:"
    echo "$remaining_large"
fi

echo ""
echo "📋 Important Notes"
echo "-----------------"
echo "✅ Git history is preserved - no commits were modified"
echo "✅ Large files can be recovered from previous commits if needed"
echo "✅ Local file copies are preserved (only removed from git tracking)"
echo "✅ .gitignore updated to prevent accidental re-adding"
echo ""
echo "💡 Next steps:"
echo "   - Move large files to external storage (Git LFS, cloud storage, etc.)"
echo "   - Update documentation with data access instructions"
echo "   - Consider using Git LFS for future large files: git lfs track '*.csv'"
echo ""
echo "🎉 Cleanup complete!"