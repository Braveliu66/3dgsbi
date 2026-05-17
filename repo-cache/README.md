# Upstream Repository Cache

Preview algorithms are bundled under `backend/app/preview/vendor`.

Fine reconstruction now uses the embedded trainer under:

```text
worker/trainer/dash_deblur_group_gs/
```

The worker image copies that trainer to:

```text
/opt/dash_deblur_group_gs
```

`repo-cache/DashDeblurGroupGS` is kept only as an optional local override location
for experiments. Set `DASH_DEBLUR_GROUP_REPO` or task option `fine_trainer_repo`
when intentionally testing a different merged trainer checkout.

CUDA extension sources for the embedded trainer live under
`worker/trainer/dash_deblur_group_gs/submodules/`. After a fresh clone, initialize
submodules before building the worker image:

```powershell
git submodule update --init --recursive
```
