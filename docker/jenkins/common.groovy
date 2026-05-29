// Shared helpers loaded by both Jenkinsfile (build pipeline) and Jenkinsfile_HW.
// Helpers that genuinely diverge between the two pipelines (e.g. safeStash,
// the directory-clean trio cleanPreviousBuildFiles / cleanBoardWorkdirHw /
// cleanReportsDirHw) expose distinct entry points instead of being
// parameterised here.

boolean paramBool(String name) {
  def v = params.get(name)
  if (v == null) { return false }
  if (v instanceof Boolean) { return v }
  return v.toString().toBoolean()
}

String paramString(String name) {
  def v = params.get(name)
  return v == null ? '' : v.toString()
}

// Sole shell-quoting primitive for the build and HW pipelines. Wraps the
// argument in single quotes and escapes embedded single quotes via the
// canonical '"'"' dance. Re-run any shell-injection tests after touching it.
String shellQuote(String s) {
  return "'" + (s ?: '').replace("'", "'\"'\"'") + "'"
}

// Sets FINN_DOCKER_PREBUILT=1 when a shared image is configured so non-builder
// agents load the image from NFS instead of rebuilding.
void runDockerCommand(String command) {
  if (env.FINN_DOCKER_SHARED_IMAGE_DIR) {
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

// Build pipeline stashes the full per-shard report sidecar set. Some are
// missing when a shard fails early, so allowEmpty is true. The .coverage
// entry only exists on rows that opted into coverage in STAGES.
void safeStashShardReport(String stashName) {
  catchError(buildResult: null, stageResult: null,
             message: "safeStashReport(${stashName}) failed, aggregation may be partial") {
    stash name: stashName,
          includes: "${stashName}.xml,${stashName}.html,${stashName}.timings.json," +
                    "${stashName}.shardmap.txt,${stashName}.shardmap.json,${stashName}.stagemap," +
                    "${stashName}.empty-shard,${stashName}.coverage",
          allowEmpty: true
  }
}

// HW pipeline files reports under ${testType}_hw_${board} but stashes them
// keyed on ${testType}_${board}, so fileBase is passed explicitly.
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

// Hard-fail on root-owned residue. Factored out so the build and HW forms
// below cannot diverge on the error message or detection logic.
void _assertNoResidue(String caller, String q) {
  sh """
    if [ -d ${q} ]; then
      echo "${caller}: ${q} still exists after rm. Likely root-owned residue. Ask an admin to 'sudo rm -rf' the directory on this agent."
      ls -la ${q} | head -40
      exit 1
    fi
  """
}

// Build pipeline form: tolerant rm, hard-fail on root-owned residue, then
// pre-create as the unprivileged user so docker -v does not bind the mount as root.
void cleanPreviousBuildFiles(String buildDir) {
  if (!buildDir || buildDir.empty) { return }
  String q = shellQuote(buildDir)
  sh "rm -rf ${q} 2>/dev/null || true"
  _assertNoResidue('cleanPreviousBuildFiles', q)
  sh "mkdir -p ${q}"
}

// HW per-board workdir form: rm both the build dir and its sibling .zip
// (the unstashed bitstream zip the next shard would otherwise overwrite
// without rotation). Sudo is added only when HW credentials are bound,
// because board agents can leave root-owned residue behind. Hard-fails on
// residue so a silently surviving root-owned tree cannot pollute the next
// shard.
void cleanBoardWorkdirHw(String buildDir) {
  if (!buildDir || buildDir.empty) { return }
  String prefix = env.USER_CREDENTIALS ? 'echo "$USER_CREDENTIALS_PSW" | sudo -S ' : ''
  String q = shellQuote(buildDir)
  String qZip = shellQuote(buildDir + '.zip')
  sh "${prefix}rm -rf ${q} ${qZip}"
  _assertNoResidue('cleanBoardWorkdirHw', q)
}

// HW reports-dir form: rm the dir only. No sibling .zip semantics, because
// the HW aggregator never produces a reports.zip alongside reports/. Sudo
// when credentials are bound for symmetry with cleanBoardWorkdirHw.
void cleanReportsDirHw(String dir) {
  if (!dir || dir.empty) { return }
  String prefix = env.USER_CREDENTIALS ? 'echo "$USER_CREDENTIALS_PSW" | sudo -S ' : ''
  String q = shellQuote(dir)
  sh "${prefix}rm -rf ${q}"
  _assertNoResidue('cleanReportsDirHw', q)
}

// All shared NFS subtrees derive from FINN_CI_NFS_ROOT. Returning '' from any
// resolver means "no NFS available". Callers must handle that as a fallback.
String finnCiNfsRoot() { return (env.FINN_CI_NFS_ROOT ?: '').trim() }

String finnSubdir(String... segments) {
  String r = finnCiNfsRoot()
  if (!r) { return '' }
  for (int i = 0; i < segments.length; i++) {
    if (!segments[i]) { return '' }
  }
  return ([r] + (segments as List)).join('/')
}

String finnAgentCachesDir(String node)    { return finnSubdir('agent_caches', node) }
String finnDockerImagesRoot()             { return finnSubdir('docker_images') }
String finnDockerImagesDir(String jobKey) { return finnSubdir('docker_images', jobKey) }
String finnArtifactsRoot()                { return finnSubdir('artifacts') }
String finnCiStateRoot()                  { return finnSubdir('_ci_state') }
String finnCiStateDir(String jobKey)      { return finnSubdir('_ci_state', jobKey) }

// Append `value` to the list at `map[key]`, creating the list lazily.
// Replaces the inline `Map.computeIfAbsent` idiom because CPS does not
// reliably transform SAM closures to java.util.function.Function. Returns
// the (possibly newly created) list so callers can chain.
List mapAppend(Map map, Object key, Object value) {
  def existing = map.get(key)
  if (existing == null) {
    existing = []
    map.put(key, existing)
  }
  existing << value
  return existing
}

return this
