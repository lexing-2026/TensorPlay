@echo off
:: Windows wheel build entry (cmd-native). Debug builds: set DEBUG=1.
:: Delegates to the bash orchestrator, which chains the MSVC env capture,
:: the build-dependency install, and the wheel build itself.
::
:: Usage: build_tensorplay.bat [output_dir]

setlocal
if "%DEBUG%" == "1" (
  set BUILD_TYPE=debug
) ELSE (
  set BUILD_TYPE=release
)
echo build_tensorplay.bat: BUILD_TYPE=%BUILD_TYPE%

bash "%~dp0build.sh" %*
if errorlevel 1 goto fail

exit /b 0

:fail
exit /b 1
