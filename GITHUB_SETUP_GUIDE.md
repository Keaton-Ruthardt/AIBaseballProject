# GitHub Setup Guide for Baseball Analytics Team

## Table of Contents
1. [What is GitHub and Why Are We Using It?](#what-is-github-and-why-are-we-using-it)
2. [First-Time Git Setup](#first-time-git-setup)
3. [Basic Workflow](#basic-workflow)
4. [Common Commands Cheat Sheet](#common-commands-cheat-sheet)
5. [Troubleshooting Common Issues](#troubleshooting-common-issues)
6. [Team Workflow and Best Practices](#team-workflow-and-best-practices)

---

## What is GitHub and Why Are We Using It?

### What is GitHub?
GitHub is a web-based platform that uses Git for version control. Think of it as a sophisticated "track changes" system for code that allows multiple people to work on the same project simultaneously without overwriting each other's work.

### Why We're Using It for Our Baseball Project
- **Collaboration**: All 5 team members (Keaton, Diego, Joshua, DuoDuo, Samuel) can work on different parts of the project simultaneously
- **Version Control**: Track all changes made to our code, data, and analysis over the 5-week timeline
- **Code Review**: Team members can review each other's work before merging it into the main project
- **Project Management**: Track tasks, issues, and progress on deliverables
- **Backup**: All work is stored in the cloud, preventing data loss
- **Transparency**: Everyone can see what others are working on and the project's current state

---

## First-Time Git Setup

### Step 1: Install Git

**Windows:**
1. Download Git from [https://git-scm.com/download/win](https://git-scm.com/download/win)
2. Run the installer with default settings
3. Verify installation by opening Command Prompt or Git Bash and typing:
   ```bash
   git --version
   ```

**Mac:**
1. Open Terminal
2. Install via Homebrew (recommended):
   ```bash
   brew install git
   ```
   Or download from [https://git-scm.com/download/mac](https://git-scm.com/download/mac)

**Linux:**
```bash
sudo apt-get update
sudo apt-get install git
```

### Step 2: Configure Git

Open your terminal/command prompt and set your identity:

```bash
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"
```

**Important**: Use the same email associated with your GitHub account.

### Step 3: Set Up SSH Keys (Recommended)

SSH keys allow you to connect to GitHub without typing your password every time.

1. **Generate SSH Key:**
   ```bash
   ssh-keygen -t ed25519 -C "your.email@example.com"
   ```
   Press Enter to accept the default file location. You can optionally set a passphrase.

2. **Add SSH Key to SSH Agent:**
   ```bash
   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ```

3. **Add SSH Key to GitHub:**
   - Copy your public key:
     ```bash
     cat ~/.ssh/id_ed25519.pub
     ```
   - Go to GitHub → Settings → SSH and GPG keys → New SSH key
   - Paste your key and save

4. **Test Connection:**
   ```bash
   ssh -T git@github.com
   ```
   You should see: "Hi [username]! You've successfully authenticated..."

---

## Basic Workflow

### 1. Clone the Repository (First Time Only)

Get a copy of the project on your computer:

```bash
cd C:\Users\[YourName]\Documents
git clone git@github.com:[organization]/AIBaseballProject.git
cd AIBaseballProject
```

### 2. Create a Branch

**NEVER work directly on the `main` branch!** Always create a feature branch:

```bash
# Create and switch to a new branch
git checkout -b [name]/your-task-name

# Example:
git checkout -b keaton/deliverable1
git checkout -b keaton/deliverable2
```

**Branch Naming Convention:**
- `[name]/[description]` - For main working branch for each member

### 3. Make Changes

Work on your files, add code, analyze data, etc.

### 4. Check Status

See what files you've changed:

```bash
git status
```

### 5. Stage Changes

Add files you want to commit:

```bash
# Stage specific files
git add path/to/file.py

# Stage all changed files
git add .

# Stage all Python files
git add *.py
```

### 6. Commit Changes

Save your changes with a descriptive message:

```bash
git commit -m "Add sacrifice play prediction model for Question 1"
```

**Good commit messages:**
- "Implement computer vision pipeline for sacrifice detection"
- "Fix data preprocessing bug in player analysis"
- "Add documentation for contract analysis methodology"


### 7. Push to GitHub

Upload your branch to GitHub:

```bash
# First time pushing a new branch
git push -u origin [name]/your-task-name

# Subsequent pushes
git push
```

### 8. Create a Pull Request (PR)

1. Go to the repository on GitHub
2. Click "Pull requests" → "New pull request"
3. Select your branch
4. Fill out the PR template with:
   - What changes you made
   - Which question it relates to
   - How to test it
   - Screenshots (if applicable)
5. Request a review from at least one team member
6. Wait for approval before merging

### 9. Update Your Local Repository

Before starting new work, always pull the latest changes:

```bash
# Switch to main branch
git checkout main

# Get latest changes
git pull origin main

# Create new branch from updated main
git checkout -b feature/new-task
```

---

## Common Commands Cheat Sheet

### Getting Started
```bash
git clone [url]                    # Copy repository to your computer
git init                           # Initialize a new Git repository
```

### Branching
```bash
git branch                         # List all branches
git branch [branch-name]           # Create new branch
git checkout [branch-name]         # Switch to branch
git checkout -b [branch-name]      # Create and switch to new branch
git branch -d [branch-name]        # Delete branch (locally)
git push origin --delete [branch]  # Delete branch (remotely)
```

### Making Changes
```bash
git status                         # Check status of files
git add [file]                     # Stage specific file
git add .                          # Stage all changes
git commit -m "[message]"          # Commit with message
git commit --amend                 # Modify last commit
```

### Syncing
```bash
git pull                           # Fetch and merge changes
git pull origin main               # Pull from main branch
git push                           # Upload commits
git push -u origin [branch]        # Push new branch to remote
git fetch                          # Download changes without merging
```

### Viewing History
```bash
git log                            # View commit history
git log --oneline                  # Condensed commit history
git log --graph --all              # Visual branch history
git diff                           # Show unstaged changes
git diff --staged                  # Show staged changes
```

### Undoing Changes
```bash
git reset [file]                   # Unstage file
git reset --hard HEAD              # Discard all local changes
git revert [commit-hash]           # Create new commit that undoes changes
git checkout -- [file]             # Discard changes to file
```

### Stashing (Temporarily Save Work)
```bash
git stash                          # Save changes temporarily
git stash list                     # List all stashes
git stash pop                      # Apply and remove latest stash
git stash apply                    # Apply stash without removing it
```

---

## Troubleshooting Common Issues

### Issue 1: "Permission Denied (publickey)"

**Problem**: Can't push to GitHub.

**Solution**:
1. Check if SSH key is set up correctly:
   ```bash
   ssh -T git@github.com
   ```
2. If it fails, review [Step 3: Set Up SSH Keys](#step-3-set-up-ssh-keys-recommended)
3. Alternatively, use HTTPS instead of SSH

### Issue 2: Merge Conflicts

**Problem**: Git can't automatically merge your changes with someone else's.

**Solution**:
1. Don't panic! This is normal.
2. Open the conflicted files (Git will mark them)
3. Look for conflict markers:
   ```
   <<<<<<< HEAD
   Your changes
   =======
   Their changes
   >>>>>>> branch-name
   ```
4. Edit the file to keep the correct version
5. Remove the conflict markers
6. Stage and commit:
   ```bash
   git add [resolved-file]
   git commit -m "Resolve merge conflict in [file]"
   ```

### Issue 3: Committed to Wrong Branch

**Problem**: Made commits on `main` instead of a feature branch.

**Solution**:
```bash
# Create a new branch with your changes
git branch feature/your-task-name

# Reset main to remote state
git checkout main
git reset --hard origin/main

# Switch to your new branch
git checkout feature/your-task-name
```

### Issue 4: Need to Undo Last Commit

**Problem**: Committed too early or made a mistake.

**Solution**:
```bash
# Keep changes but undo commit
git reset --soft HEAD~1

# Discard changes and undo commit
git reset --hard HEAD~1
```

### Issue 5: Accidentally Deleted Files

**Problem**: Deleted files you didn't mean to.

**Solution**:
```bash
# Restore specific file
git checkout HEAD -- [file]

# Restore all deleted files
git checkout HEAD -- .
```

### Issue 6: "Your branch is behind 'origin/main'"

**Problem**: Your local repository is outdated.

**Solution**:
```bash
git pull origin main
```

### Issue 7: Large Files Won't Push

**Problem**: Error about file size limits.

**Solution**:
1. Don't commit large data files or model files
2. Use `.gitignore` to exclude them
3. Store large files elsewhere (Google Drive, cloud storage)
4. If already committed, remove from history:
   ```bash
   git rm --cached [large-file]
   git commit -m "Remove large file"
   ```

---

### Pull Request Best Practices

1. **Before Creating a PR:**
   - Test your code thoroughly
   - Make sure it runs without errors
   - Update documentation if needed
   - Pull latest changes from main and resolve conflicts

2. **PR Description Should Include:**
   - Summary of changes
   - Related issue number (e.g., "Closes #15")
   - Testing steps
   - Screenshots (for visualizations/UI)
   - Any breaking changes


### Commit Best Practices

- **Commit often**: Small, focused commits are better than large ones
- **Write clear messages**: Others should understand what changed without reading the code
- **One logical change per commit**: Don't mix unrelated changes
- **Test before committing**: Don't commit broken code

### Branch Management

- **Keep branches short-lived**: Merge within 1-3 days
- **Delete merged branches**: Keeps repository clean
- **Sync with main regularly**: Pull from main into your feature branch
- **One branch per task**: Don't work on multiple unrelated tasks in one branch


### What NOT to Commit

Add these to `.gitignore`:
- Large data files (> 50 MB)
- Trained model files
- Virtual environment folders (`venv/`, `env/`)
- IDE-specific files (`.vscode/`, `.idea/`)
- Temporary files (`*.pyc`, `__pycache__/`)
- API keys or credentials
- Personal notes

### Communication

- **Use Issues** for task tracking and bug reports
- **Use PR comments** for code-specific discussions
- **Use project board** for high-level progress tracking
- **Tag team members** (@username) when you need their input
- **Link issues in commits**: Use "Closes #5" in commit messages

### Emergency Contact

If you're completely stuck:
1. Don't force push (`git push --force`) unless you know what you're doing
2. Don't delete the repository
3. Ask for help in the team chat
4. Create an issue describing your problem
5. Contact the team leads (Keaton, Joshua)

---

## Quick Start for Complete Beginners

If you've never used Git before, follow these steps to get started:

1. **Install Git** (15 minutes)
   - Download and install from git-scm.com
   - Open terminal and run: `git --version`

2. **Configure Git** (5 minutes)
   ```bash
   git config --global user.name "Your Name"
   git config --global user.email "your.email@example.com"
   ```

3. **Clone the Repository** (5 minutes)
   ```bash
   cd Documents
   git clone [repository-url]
   cd baseball-analytics
   ```

4. **Create Your First Branch** (2 minutes)
   ```bash
   git checkout -b [name]/my-first-branch
   ```

5. **Make a Change** (10 minutes)
   - Edit a file or create a new one
   - Save it

6. **Commit Your Change** (5 minutes)
   ```bash
   git add .
   git commit -m "My first commit"
   git push -u origin feature/my-first-branch
   ```

7. **Create a Pull Request** (10 minutes)
   - Go to GitHub
   - Click "Pull requests" → "New pull request"
   - Select your branch and create PR

**Total time: ~1 hour to go from zero to your first PR!**

---

## Additional Resources

- [Official Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Interactive Git Tutorial](https://learngitbranching.js.org/)
- [Git Cheat Sheet PDF](https://education.github.com/git-cheat-sheet-education.pdf)
- [Visualizing Git](https://git-school.github.io/visualizing-git/)

