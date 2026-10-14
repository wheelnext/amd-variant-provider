# AMD Variant Provider Plugin

This is a plugin for the proposed [wheel variants implementation](
https://github.com/wheelnext/pep_xxx_wheel_variants) that provides
properties specific to AMD GPUs and the installed ROCm stack.

The provider is dependency-free. It does not import `pyrsmi` (the
`pyrsmi.py` module in this repository is unused). It does not open
network connections, and it does not invoke a shell: every external
command is a fixed argument list.

## Usage

Namespace: `amd`

Plugin API: `amd_variant_provider` (the default)

Example use in `pyproject.toml`:

```toml
[variant.providers.amd]
requires = ["amd-variant-provider"]
```

## Provided properties

To obtain the full list of properties supported by a given plugin
version, install it, then use:

```sh
variantlib plugins get-configs -a -n amd
```

To obtain the full list of properties compatible with your system, use:

```sh
variantlib plugins get-configs -s -n amd
```

`get_supported_configs()` emits properties in this order. A property is
omitted when detection finds nothing and no override is set.
`kmd_version` is probed (see below) but is not emitted.

### amd :: gfx_arch

Values: `gfx900`, `gfx906`, `gfx908`, `gfx90a`, `gfx942`, `gfx1030`,
`gfx1100`, `gfx1101`, `gfx1102`, `gfx1200`, `gfx1201`

Multi-value: yes. At install time every accepted architecture is
emitted together, in one property.

A wheel sets this when it is built for specific AMD GPU architectures
(GFX names such as `gfx942` for Instinct MI300 or `gfx1100` for RDNA3).
The wheel will not be selected on a system whose emitted list does not
include that architecture.

`get_all_configs()` publishes the values above. A forced or detected
name outside that list can still appear in `get_supported_configs()`.

### amd :: rocm_version

Values: `7.10`, `7.9`, ... `7.0`, `6.4`, `6.3`

Multi-value: no. At install time the plugin emits one `major.minor`
string. The patch component is detected when present and then dropped.

A wheel sets this to the ROCm user-space version it was built for.
The wheel will not be selected when the emitted version differs.

`get_all_configs()` publishes the values above, newest first.
A detected version outside that list can still appear in
`get_supported_configs()`.

## Detection

`get_system_info()` gathers GFX, the ROCm version, and the kernel-mode
driver version in one pass. That pass runs when either emitted property
is left to detection. If both [overrides](#overrides) are set, no file
is read and no subprocess is started.

`rocminfo` is skipped unless `platform.system()` is `Linux`. The version
file and `rocm_agent_enumerator` are not gated on the operating system.

### Interfaces

| Interface | When it is used | What is read |
|-----------|-----------------|--------------|
| `rocminfo` | The system probe runs, the OS is Linux, and `rocminfo` is on `PATH` | Subprocess with no arguments. Stdout is parsed for GFX names and for the ROCk module version. |
| `$ROCM_PATH/.info/version`, or `/opt/rocm/.info/version` when `ROCM_PATH` is unset | The system probe runs | File read. The file contains `X.Y` or `X.Y.Z` (a `-` suffix is ignored). This is the only source of `rocm_version`; `rocminfo` is not parsed for it. |
| `rocm_agent_enumerator -name` | The system probe runs, GFX was not accepted from `rocminfo`, and the executable is on `PATH` | Subprocess. Stdout is a list of architecture names. |
| `/sys/module/amdgpu/version` | The system probe runs and `rocminfo` did not report a ROCk version | File read. Not emitted. |
| `modinfo amdgpu` | That sysfs file is missing, empty, or unreadable | Subprocess. The `version:` line is parsed as `X.Y.Z`. Not emitted. |

### amd :: gfx_arch

1. On Linux, if `rocminfo` is on `PATH`, run it and collect tokens
   matching `gfx` plus digits and hexadecimal characters (for example
   `gfx942`). The match is case-sensitive. Sort and deduplicate them.
   Accept the set only when every token is on the allowlist. The allowlist is
   `AMD_PREFERRED_GFX_ARCHS` when that variable is a non-empty
   comma-separated list; otherwise it is the value list in
   [amd :: gfx_arch](#amd--gfx_arch). If any token is off the allowlist,
   the whole `rocminfo` result is discarded.
2. If that did not yield a list, and `rocm_agent_enumerator` is on
   `PATH`, run it with `-name`. Keep the same kind of token, lowercase
   it, drop `gfx000` (that token is the CPU agent), then sort and
   deduplicate. This path has no allowlist.

### amd :: rocm_version

Read `$ROCM_PATH/.info/version`, or `/opt/rocm/.info/version` when
`ROCM_PATH` is unset. Split the contents on `.` or `-`. Require at least
`major.minor`. Emit `major.minor`.

### kmd_version (not emitted)

Kept for diagnostics inside `get_system_info()`. Reimplementations do
not need to publish it.

1. If `rocminfo` ran, parse `ROCk module version X.Y.Z` from its stdout.
2. Otherwise read `/sys/module/amdgpu/version` and split on `.`.
3. Otherwise run `modinfo amdgpu` and parse the `version:` line as
   `X.Y.Z`.

On a kernel where `amdgpu` is built in, the sysfs version file is often
absent and `modinfo` has no module version, so this probe returns
nothing. That does not affect the emitted properties.

## Overrides

These environment variables replace the emitted value for one property.
They do not by themselves skip the system probe: the probe is skipped
only when both variables are set. `AMD_VARIANT_PROVIDER_FORCE_GFX_ARCH`
is split on commas, whitespace, or semicolons.

| Variable | Effect |
|----------|--------|
| `AMD_VARIANT_PROVIDER_FORCE_GFX_ARCH` | Emit these GFX names instead of the detected list. Example: `gfx1100,gfx1101`. |
| `AMD_VARIANT_PROVIDER_FORCE_ROCM_VERSION` | Must be `major.minor.patch`. Emit `major.minor` instead of the version-file value. Example: `7.0.0`. |
