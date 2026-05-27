// Shared utilities for docker/jenkins/Jenkinsfile and docker/jenkins/Jenkinsfile_HW.
// Loaded via load() in each Jenkinsfile once the SCM checkout has dropped the
// repo onto the active node, then accessed through thin top-level wrappers in
// the calling Jenkinsfile so existing call sites read the same as before.
//
// Adding a helper here keeps both pipelines from drifting. Helpers that are
// genuinely pipeline-specific (e.g. PARALLEL_SHARDS, HW_SHARDS, the various
// printLsfSummary helpers) stay in their owning Jenkinsfile.

// PRESET overrides the CI-progression booleans supplied by the SW Jenkinsfile.
// The HW Jenkinsfile has no PRESET param, so the branch is skipped there.
// ``smokeParams`` is the explicit SMOKE_PARAMS list from ci_sharding.py
// loaded once at Validate time, never derived from CI_PARAMS ordering.
boolean paramBool(String name, Collection ciProgressionParams = [], Collection smokeParams = []) {
  def presetValue = params.get('PRESET')
  def ciParams = (ciProgressionParams ?: []) as List
  if (presetValue != null && name in ciParams) {
    String preset = presetValue.toString().toLowerCase().trim() ?: 'custom'
    if (preset == 'smoke') { return (smokeParams ?: []).contains(name) }
    if (preset == 'full')  { return true }
  }
  def v = params.get(name)
  if (v == null) { return false }
  if (v instanceof Boolean) { return v }
  return v.toString().toBoolean()
}

String paramString(String name) {
  def v = params.get(name)
  return v == null ? '' : v.toString()
}

String shellQuote(String s) {
  return "'" + (s ?: '').replace("'", "'\"'\"'") + "'"
}

// Hints run-docker.sh with FINN_DOCKER_PREBUILT=1 when a shared image is
// configured so non-builder agents docker-load from NFS instead of rebuilding.
// Falls through to a plain sh when no shared image is configured.
void runDockerCommand(String command) {
  if (env.FINN_DOCKER_SHARED_IMAGE_DIR || env.FINN_DOCKER_SHARED_DIR) {
    withEnv(['FINN_DOCKER_PREBUILT=1']) {
      sh command
    }
  } else {
    sh command
  }
}

void unstashIfPresent(String stashName) {
  try {
    unstash stashName
  } catch (Exception ignored) {
    echo "No stash '${stashName}' (stage skipped or failed before publishing)"
  }
}

// SW form: stashes the full per-shard report sidecar set (xml, html,
// timings.json, shardmap.*, stagemap). allowEmpty=true because not every
// shard produces every sidecar.
void safeStashShardReport(String stashName) {
  catchError(buildResult: null, stageResult: null,
             message: "safeStashReport(${stashName}) failed, aggregation may be partial") {
    stash name: stashName,
          includes: "${stashName}.xml,${stashName}.html,${stashName}.timings.json," +
                    "${stashName}.shardmap.txt,${stashName}.shardmap.json,${stashName}.stagemap",
          allowEmpty: true
  }
}

// HW form: stash only when the JUnit XML exists, with a different filename
// pattern from the stashName (HW tests file their reports under
// "${testType}_hw_${board}" while the stash is keyed on "${testType}_${board}").
void safeStashHwReport(String stashName, String fileBase) {
  catchError(buildResult: null, stageResult: null,
             message: "safeStashReport(${stashName}) failed, aggregation may be partial") {
    if (fileExists("${fileBase}.xml")) {
      stash name: stashName,
            includes: "${fileBase}.xml,${fileBase}.html",
            allowEmpty: false
    }
  }
}

// SW form: rm with tolerance, hard-fail on root-owned residue, then mkdir.
// Pre-create as the unprivileged user so docker's -v doesn't bind as root.
void cleanPreviousBuildFilesStrict(String buildDir) {
  if (!buildDir || buildDir.empty) { return }
  String q = shellQuote(buildDir)
  sh """
    rm -rf ${q} 2>/dev/null || true
    if [ -d ${q} ]; then
      echo "cleanPreviousBuildFiles: ${q} still exists after rm. Likely root-owned residue. Ask an admin to 'sudo rm -rf' the directory on this agent."
      ls -la ${q} | head -40
      exit 1
    fi
    mkdir -p ${q}
  """
}

// HW form: rm with optional sudo when HW credentials are bound, and an
// optional sibling sweep (e.g. cleaning KV260_SOM/ also sweeps KV260_SOM.zip).
// Targets are enumerated explicitly so the next maintainer does not have to
// derive the behaviour from glob-expansion semantics.
void cleanPreviousBuildFilesHw(String buildDir, boolean includeSiblings) {
  if (!buildDir || buildDir.empty) { return }
  String prefix = env.USER_CREDENTIALS ? 'echo "$USER_CREDENTIALS_PSW" | sudo -S ' : ''
  String targets = includeSiblings
      ? "${shellQuote(buildDir)} ${shellQuote(buildDir + '.zip')}"
      : shellQuote(buildDir)
  sh "${prefix}rm -rf ${targets}"
}

// FINN_CI_NFS_ROOT consolidates the legacy FINN_NFS_ROOT_BASE,
// FINN_DOCKER_SHARED_DIR, and ARTIFACT_DIR env vars into one. The legacy
// vars still win when explicitly set so a partial migration cannot relocate
// any tree silently. Conventional subdir layout under FINN_CI_NFS_ROOT:
//   agent_workspaces/<NODE>/workspace/tmp     (legacy FINN_NFS_ROOT_BASE)
//   docker_images/<jobKey>/<BUILD>/           (legacy FINN_DOCKER_SHARED_DIR)
//   artifacts/ci_runs/<jobKey>/<BUILD>/       (legacy ARTIFACT_DIR)
String finnCiNfsRoot()       { return (env.FINN_CI_NFS_ROOT ?: '').trim() }
String finnNfsRootBase()     { return resolveNfsTree('FINN_NFS_ROOT_BASE',     'agent_workspaces') }
String finnDockerSharedDir() { return resolveNfsTree('FINN_DOCKER_SHARED_DIR', 'docker_images') }
String finnArtifactDir()     { return resolveNfsTree('ARTIFACT_DIR',           'artifacts') }

String resolveNfsTree(String legacyEnv, String subdir) {
  String legacy = legacyNfsEnv(legacyEnv)
  if (legacy) { return legacy }
  String root = finnCiNfsRoot()
  if (!root) { return '' }
  return "${root}/${subdir}"
}

String legacyNfsEnv(String name) {
  if (name == 'FINN_NFS_ROOT_BASE')     { return (env.FINN_NFS_ROOT_BASE ?: '').trim() }
  if (name == 'FINN_DOCKER_SHARED_DIR') { return (env.FINN_DOCKER_SHARED_DIR ?: '').trim() }
  if (name == 'ARTIFACT_DIR')           { return (env.ARTIFACT_DIR ?: '').trim() }
  error "legacyNfsEnv: unknown env var ${name}"
}

// Echoes once per build when FINN_CI_NFS_ROOT is set alongside any of the
// legacy vars, so operators know which path actually won.
void warnLegacyNfsEnv() {
  if (!finnCiNfsRoot()) { return }
  warnLegacyNfsEnvOne('FINN_NFS_ROOT_BASE',     env.FINN_NFS_ROOT_BASE)
  warnLegacyNfsEnvOne('FINN_DOCKER_SHARED_DIR', env.FINN_DOCKER_SHARED_DIR)
  warnLegacyNfsEnvOne('ARTIFACT_DIR',           env.ARTIFACT_DIR)
}

void warnLegacyNfsEnvOne(String name, String value) {
  if ((value ?: '').trim()) {
    echo "warnLegacyNfsEnv: ${name} is set alongside FINN_CI_NFS_ROOT, the legacy var still wins for safety. Drop it once you have verified the FINN_CI_NFS_ROOT-derived layout."
  }
}

return this
