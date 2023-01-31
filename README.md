# PROJECT_TEMPLATE

This is our template for a hybrid github/dropbox project structure, which
enables us to keep large files in sync between machines via dropbox, while
using version control to manage projects via github.

By default, large files (.pth.tar, .tar.gz, .zip, etc.), all pdfs, and all image, video, and audio assests are ignored (with the exception of images stored in the images folder).

We also use nbstripout to prevent notebook cell outputs from getting committed to version control.

Finally, we use a lightweight "dvc.py" utility to track large files via git without commiting them.
e.g., if you run

```
python dvc.py track_files --pattern '**/*.pth.tar'
```

Then all .pth.tar files will be hashed, with fileinfo (e.g., filename, hashid) stored in a .dvc file (the ".dvc" is borrowed from "data version control").

# getting started

- [ ] goto https://github.com/harvard-visionlab/PROJECT_TEMPLATE, click "use this template"
- [ ] use the nameing convension PROJECT_SHORTNAME, e.g. PROJECT_UNETS

- [ ] clone to first machine's DropboxProjects folder, renaming to short name (drop "PROJECTS\_")

```
cd /home/jovyan/work/DropboxProjects
git clone --separate-git-dir=/home/jovyan/work/.separate_gitroots/PROJECT_SHORTNAME.git git@github.com:harvard-visionlab/PROJECT_SHORTNAME.git SHORTNAME
```

- [ ] clone to second machine's GITHUB folder, so the synced repo will still point to the correct 'separate-git-dir'

```
cd /home/jovyan/work/Github
git clone --separate-git-dir=/home/jovyan/work/.separate_gitroots/PROJECT_SHORTNAME.git git@github.com:harvard-visionlab/PROJECT_SHORTNAME.git
```

- [ ] run the install script on both machines (installs the project and nbstripout; more about these steps below)

```
chmod u+x install.sh
./install.sh
```

That's it! Large files will be synced via dropbox. Code (.py, .ipynb without cell ouputs) and .dvc files will be tracked via github.

When new files are added on one machine, you'll have to "git add ." them on other machines before you can make pull requests.

## make changes on machine #1

```
git add .
git commit -m "add stuff"
git push origin main
```

## sync changes to machine #2, retaining notebook cell outputs (via dropbox)

```
git add . && git pull
```

## sync changes across devboxes stashing local changes

```
git add . && git stash && git pull
```

# setup nbstripout (already done if you run ./install.sh)

We use [nbstripout](https://www.youtube.com/watch?v=BEMP4xacrVc) to prevent notebook cell outputs from getting committed to version control. The cell outputs will remain on disk but are filtered by nbstrip.

Make sure nbstripout is installed

```
pip install nbstripout nbconvert
```

Then within this repo, run the following to install the hooks that do the filtering:

```
nbstripout --install
```

## install the project (already done if you run ./install.sh)

Installing the project enables us to import from PROJECT_SHORTNAME from anywhere, avoiding annoying relative import issues.

```
pip install --user -e .
```
