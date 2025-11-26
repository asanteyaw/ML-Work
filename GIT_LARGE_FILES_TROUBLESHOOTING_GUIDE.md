# Git Large Files Troubleshooting Guide
**How to Fix GitHub's "Large files detected" Error**

*Step-by-step solution used to fix the ML-Work repository on 2025-11-25*

## 🚨 The Problem

When trying to push to GitHub, you get this error:
```
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
To https://github.com/asanteyaw/ML-Work.git
 ! [remote rejected] main -> main (pre-receive hook declined)
error: failed to push some refs to 'https://github.com/asanteyaw/ML-Work.git'
```

**Key Point**: Even after removing large files from your working directory, they remain in Git's history, and GitHub checks the entire repository history.

## 🔍 Step 1: Diagnose the Problem
/Users/yawasante/Documents/Doctrate/Thesis/Python/ML-Work/01_vol_nn_integration

### Check Current Repository Size
```bash
cd /Users/yawasante/Documents/Doctrate/Thesis/Python/ML-Work
du -sh .
```
**Output**: `579M` (way too large!)

### Check Git Status
```bash
git status
```
**Output**: 
```
On branch main
Your branch is ahead of 'origin/main' by 5 commits.
Changes not staged for commit:
    modified:   .DS_Store
    deleted:    01_vol_nn_integration/result_tables/.DS_Store
```

### Find Current Large Files (>50MB)
```bash
find . -type f -size +50M | head -10
```
**Output**: 
```
./.git/objects/pack/pack-b4ec1292011ee688fb6e354c08d435df68bedb05.pack
```

The large files are in Git's internal storage, not your working directory!

## 🕵️ Step 2: Find Large Files in Git History

This is the crucial diagnostic command:
```bash
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort --numeric-sort --key=2 | tail -10
```

**Output** (the smoking gun):
```
1646d861af382af558e450985d0ee1e074ef7e98 20161735 01_vol_nn_integration/all_data/R.pt
e4108951e85827490afb2a05597962af421e0ed3 20161743 01_vol_nn_integration/all_data/CR.pt
17be7349833a1701869b6e7074ede2fb6742bdf8 20241735 01_vol_nn_integration/all_data/h.pt
b11b5f8386d38778105f28574e8add5ae458f7af 20241743 01_vol_nn_integration/all_data/Ch.pt
5dcb09fc6d691b18edf1acefc88b5eeb5983e4b6 45398306 01_vol_nn_integration/all_data/sample0.csv
846c2206df4f2c54c614d4e9fff1a6258c2f6eb9 45399319 01_vol_nn_integration/all_data/sample_1.csv
6e8aede8da83442720658ba3a950bde69bc9c309 45400496 01_vol_nn_integration/all_data/sample_3.csv
db90a64d55f2843b81e10dadfec859c96b6532eb 45401234 01_vol_nn_integration/all_data/sample_4.csv
4e7e2e9b48f12f5949372b25766bbb164aa4fb44 45401343 01_vol_nn_integration/all_data/sample_2.csv
52d7c8a453c6584f717eb26c935d2aa32fb4adab 1615376835 01_vol_nn_integration/external/utils/US_Accidents_Dec20.csv
```

**Analysis**:
- `US_Accidents_Dec20.csv`: **1.6GB** (!!!)
- `sample*.csv` files: ~45MB each
- `*.pt` files: ~20MB each

## 🛡️ Step 3: Create Preventive Measures

Before cleaning, create a `.gitignore` to prevent future issues:

```bash
cat > .gitignore << 'EOF'
# Large Data Files
*.csv
*.pt
*.pth
*.pkl
*.pickle
*.h5
*.hdf5
*.mat

# Build artifacts
build/
*.o
*.a
*.so
*.dylib
*.dll
CMakeCache.txt
CMakeFiles/
cmake_install.cmake
Makefile
compile_commands.json

# IDE and OS files
.vscode/
.DS_Store
Thumbs.db
*.swp
*.swo
*~

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.venv
pip-log.txt
pip-delete-this-directory.txt

# Jupyter Notebook
.ipynb_checkpoints

# Data directories
data/
datasets/
all_data/
external/utils/

# Model files
models/
trained_models/
checkpoints/
*.model

# Results and outputs
results/
result_tables/
losses/
outputs/
logs/
*.log

# Temporary files
tmp/
temp/
EOF
```

Commit the `.gitignore`:
```bash
git add . 
git commit -m "Add comprehensive .gitignore and clean up files"
```

## 🧹 Step 4: Clean Git History

### Install git-filter-repo (if needed)
```bash
brew install git-filter-repo
```

### Clean the Repository History
**⚠️ WARNING: This rewrites Git history - make a backup first!**

```bash
git filter-repo \
  --path 01_vol_nn_integration/external/utils/US_Accidents_Dec20.csv --invert-paths \
  --path 01_vol_nn_integration/all_data/sample0.csv --invert-paths \
  --path 01_vol_nn_integration/all_data/sample_1.csv --invert-paths \
  --path 01_vol_nn_integration/all_data/sample_2.csv --invert-paths \
  --path 01_vol_nn_integration/all_data/sample_3.csv --invert-paths \
  --path 01_vol_nn_integration/all_data/sample_4.csv --invert-paths \
  --force
```

**Success Output**:
```
NOTICE: Removing 'origin' remote; see 'Why is my origin removed?' in the manual
Parsed 9 commits
New history written in 0.32 seconds; now repacking/cleaning...
Repacking your repo and cleaning out old unneeded objects
HEAD is now at d1c463e Add comprehensive .gitignore and clean up files
Completely finished after 8.55 seconds.
```

## ✅ Step 5: Verify the Fix

Check the new repository size:
```bash
du -sh .
```
**Result**: `188M` (down from `579M` - **391MB reduction!**)

## 🔄 Step 6: Re-establish GitHub Connection

The cleaning process removes the remote, so add it back:
```bash
git remote add origin https://github.com/asanteyaw/ML-Work.git
```

## 🚀 Step 7: Push the Clean Repository

Since we completely rewrote history, we need to force push:
```bash
git push --force origin main
```

**Success Output**:
```
Enumerating objects: 835, done.
Counting objects: 100% (835/835), done.
Delta compression using up to 10 threads
Compressing objects: 100% (410/410), done.
Writing objects: 100% (835/835), 89.57 MiB | 46.51 MiB/s, done.
Total 835 (delta 381), reused 835 (delta 381), pack-reused 0
remote: Resolving deltas: 100% (381/381), done.
To https://github.com/asanteyaw/ML-Work.git
 + b3533e7...d1c463e main -> main (forced update)
```

## 🎉 Success!

Repository now successfully pushes to GitHub without errors.

## 📋 Quick Reference for Future Use

### Diagnostic Commands
```bash
# Check repo size
du -sh .

# Find large files in working directory
find . -type f -size +50M | head -10

# Find large files in Git history
git rev-list --objects --all | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' | sed -n 's/^blob //p' | sort --numeric-sort --key=2 | tail -10
```

### Cleaning Commands
```bash
# Install git-filter-repo
brew install git-filter-repo

# Clean specific files (replace with your actual file paths)
git filter-repo --path PATH/TO/LARGE/FILE.csv --invert-paths --force

# Re-add remote and push
git remote add origin https://github.com/USERNAME/REPO.git
git push --force origin main
```

## 🚨 Important Notes

1. **Backup First**: Always backup your repository before using `git filter-repo`
2. **Rewrites History**: All commit hashes change - collaborators need fresh clones
3. **Force Push Required**: The `--force` flag is necessary due to history rewrite
4. **Team Communication**: Inform all collaborators about the history rewrite
5. **Prevention is Key**: Use `.gitignore` to prevent future large file commits

## 🌟 Alternative Solutions

### Git LFS (for files you want to keep)
```bash
git lfs install
git lfs track "*.csv"
git lfs track "*.pt"
git add .gitattributes
git commit -m "Configure Git LFS"
```

### BFG Repo-Cleaner (alternative to git-filter-repo)
```bash
brew install bfg
bfg --delete-files US_Accidents_Dec20.csv
git reflog expire --expire=now --all && git gc --prune=now --aggressive
```

---

**File Created**: 2025-11-25  
**Problem**: GitHub large files error  
**Solution**: git-filter-repo cleanup + comprehensive .gitignore  
**Result**: 391MB reduction, successful GitHub push  
