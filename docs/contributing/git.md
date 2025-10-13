# Git Tutorial

CapyMOA uses git for version control and GitHub for hosting the repository. We follow a
forking workflow, which is common for open source projects. It allows anyone to
contribute changes while keeping the main repository clean and stable. If you're familiar
with git, you can safely skip this guide, as it is intended for beginners.

## Contributing Changes

### 1. Get Your Own Copy (Forking)

Before you can work on CapyMOA, you need your own copy to edit. Go to the project's
[repository on **GitHub**](https://github.com/adaptive-machine-learning/CapyMOA) and
click the **Fork** button. This creates a complete copy of the project under your own
GitHub account (your "fork"). You can make changes here without affecting the original
project.

### 2. Download Your Copy (Cloning) and Set Up Tracking
Now you'll download your new fork to your local computer and tell Git about the original
project. 
```bash
# Download your fork to your computer (this is your 'origin').
# Replace $USER with your GitHub username and $FORK_MAME with your fork name 
# (probably 'CapyMOA').
git clone https://github.com/$USER/$FORK_MAME.git
# Or using SSH (preferable if you have SSH keys set up):
# git clone git@github.com:$USER/$FORK_NAME.git

cd CapyMOA

# Add a reference to the original repository (this is the 'upstream')
git remote add upstream https://github.com/adaptive-machine-learning/CapyMOA.git
# Or using SSH:
# git remote add upstream git@github.com:adaptive-machine-learning/CapyMOA.git
```

### 3. Create a Branch
You should make changes for each feature on a dedicated 'feature' branch. A feature
branch is a set of changes that implements a single feature and can be code reviewed
reasonably quickly. The bigger your feature the harder it'll be to review. So keep them
small if at all possible.

```bash
# Switch to the 'main' branch. This will act as the parent for our feature.
git switch main

# This step ensures that our local 'main' branch is up-to-date.
# If you skip this step you may end up basing your work on an old version of the code.
# This can create conflicts and issues later on.
git pull upstream main

# Create a new branch called 'my-branch' and switch to it immediately.
# Replace `my-branch` with your feature name.
git switch --create my-branch
# Or the older way:
# git checkout -b my-branch
```

### 4. Commit Your Changes
Once you've made edits, you need to tell Git what files you changed and other developers
why.
```bash
# Stage the changes (prepare them to be committed)
git add my_change_file.py another_file.py

# Review the changes
git status

# Commit and add a message ('-m' to set message inline)
git commit -m "My message"
```

* Refer to the [installation guide](../installation.rst) to set up your development
  environment.

### 5. Upload Your Changes to Your Fork

Your changes are still only on your local computer. Now, you need to push them up to
your personal fork on GitHub. 
```bash
git push origin my-feature
```
This uploads your `my-feature` branch to your fork on GitHub, making it visible online.

### 6. Propose Your Changes (Pull Request)

Your code is now on your fork. The final step is to ask the original project maintainers
to review and merge your work. A Pull Request formally proposes your changes and opens a
discussion channel for review. 

Go to the original project's GitHub page. GitHub will usually display a banner prompting
you to **"Compare & pull request."** Click that to open a **Pull Request (PR)**. 

It helps to review the diffs to ensure you committed what you wanted to. A clean PR that
makes targeted changes is easier to review and is less likely to break things. If you
push changes to your branch they will update on GitHub.

A PR may start some automated checks of your proposed changes.  If any checks fail
you'll need to add additional commits to fix that.

### Staying Current and Resolving Conflicts (Rebasing)

Sometimes the original project's `main` branch will be updated before your changes are
accepted. If those changes conflict with yours, you'll need to update your branch.

```bash
# Fetch and apply the latest changes from upstream main onto your current branch
git pull upstream main --rebase
```
Rebasing is a clean way to update your branch. It takes all of your commits, temporarily
removes them, updates your base branch to the latest `upstream main`, and then reapplies
your commits _on top_ of the new code. This keeps your commit history clean and linear. 

You must **force** push the rebase changes to the remote repository. This is because it
must override the history. **Force pushing can overwrite remote history and cause data
loss, especially if others are working on the same branch. This overwrites history so
it's possible to lose work by doing this incorrectly.
```bash
git push --force-with-lease origin my-branch
```
git push --force-with-lease origin my-feature
(`origin/my-branch`) if it is ahead of the local branch (`my-branch`).
For example, if someone else has pushed new commits to `origin/my-branch` after you last
pulled, `git push --force-with-lease` will refuse to overwrite those changes, helping
prevent accidental data loss. This is safer than `--force`, which will overwrite the
remote branch regardless of any new commits.

## Glossary

- **Fork**: Your own copy of a project on GitHub; changes here don't affect the original.
- **Repository (Repo)**: Stores all project files, history, and revisions.
- **Clone**: Download a remote repo to your computer for local work.
- **Origin**: Remote name for your fork on GitHub; where you push changes.
- **Upstream**: Remote name for the original project; used to pull updates.
- **Branch**: Separate line of development; changes are isolated until merged.
- **Feature Branch**: Temporary branch for a specific feature or fix; deleted after merging.
- **Commit**: Snapshot of changes with a message explaining why.
- **Staging**: Prepares changes (`git add`) for the next commit.
- **Push**: Upload local commits/branches to a remote repo.
- **Pull Request (PR)**: Request to merge your changes into the original project.
- **Merge**: Integrate changes from one branch into another.
- **Rebase**: Replay your commits on top of the latest upstream code for a clean history.
- **Conflict**: When the same code is changed differently in two branches; needs manual resolution.

## Alternative Guides
Here are some alternative guides that cover forking workflows in Git:

- [GitHub Docs: Fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo)
- [Atlassian Git Tutorials](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
- [Scott's Weblog Git Forking Workflow](https://blog.scottlowe.org/2015/01/27/using-fork-branch-git-workflow/)
