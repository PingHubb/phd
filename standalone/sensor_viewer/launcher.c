/* Small native Windows GUI launcher. No installer, shell, admin rights or network. */
#define UNICODE
#define _UNICODE
#include <windows.h>
#include <wchar.h>
#include <stdio.h>

int WINAPI wWinMain(HINSTANCE instance, HINSTANCE previous, PWSTR arguments, int show) {
    wchar_t root[16000], python[16384], script[16384], command[32768];
    DWORD size = GetModuleFileNameW(NULL, root, 16000);
    if (size == 0 || size >= 15900) return 1;
    wchar_t *separator = wcsrchr(root, L'\\');
    if (!separator) return 1;
    *separator = L'\0';
    _snwprintf(python, 16384, L"%ls\\runtime\\pythonw.exe", root);
    _snwprintf(script, 16384, L"%ls\\app\\run.py", root);
    if (GetFileAttributesW(python) == INVALID_FILE_ATTRIBUTES ||
        GetFileAttributesW(script) == INVALID_FILE_ATTRIBUTES) {
        MessageBoxW(NULL, L"Extract the entire ZIP first. Keep the runtime and app folders beside PingLabSensorViewer.exe.", L"PingLab Sensor Viewer", MB_OK | MB_ICONERROR);
        return 1;
    }
    int length = _snwprintf(command, 32768, L"\"%ls\" -B \"%ls\" %ls", python, script, arguments);
    if (length < 0 || length >= 32767) return 1;
    STARTUPINFOW startup = {0};
    PROCESS_INFORMATION process = {0};
    startup.cb = sizeof(startup);
    if (!CreateProcessW(python, command, NULL, NULL, FALSE, 0, NULL, root, &startup, &process)) {
        wchar_t error[512];
        _snwprintf(error, 512, L"Unable to start the bundled runtime (Windows error %lu).\nExtract the full ZIP to a normal folder and try again. This package is for Windows x64.", GetLastError());
        MessageBoxW(NULL, error, L"PingLab Sensor Viewer", MB_OK | MB_ICONERROR);
        return 1;
    }
    CloseHandle(process.hThread);
    WaitForSingleObject(process.hProcess, INFINITE);
    DWORD result = 0;
    GetExitCodeProcess(process.hProcess, &result);
    CloseHandle(process.hProcess);
    if (result != 0) {
        MessageBoxW(NULL, L"The app exited with an error. See %LOCALAPPDATA%\\PingLabSensorViewer\\viewer.log, or run Diagnose.cmd in the extracted folder.", L"PingLab Sensor Viewer", MB_OK | MB_ICONERROR);
    }
    return (int)result;
}
