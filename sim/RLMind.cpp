
#include <Python.h>
#include <windows.h>

#include <filesystem>
#include <iostream>
#include <vector>

namespace fs = std::filesystem;

int Fail(const std::string& message) {
	std::cout << "RLMind: " << message << "\n";
	std::cout << "Press enter to close...\n";
	std::cin.get();
	return EXIT_FAILURE;
}

int wmain(int argc, wchar_t* argv[]) {
	wchar_t exePath[MAX_PATH];
	GetModuleFileNameW(nullptr, exePath, MAX_PATH);

	fs::path exeDir = fs::path(exePath).parent_path();
	fs::path root = exeDir;

	while (!fs::exists(root / "main.py")) {
		if (root == root.root_path())
			return Fail("Couldn't find main.py above " + exeDir.string());
		root = root.parent_path();
	}

	if (!fs::is_directory(root / "collision_meshes"))
		return Fail("Missing collision_meshes in " + root.string());

	bool simBuilt = false;
	for (const auto& file : fs::directory_iterator(root / "sim")) {
		auto name = file.path().filename().string();
		if (name.starts_with("_sim") && name.ends_with(".pyd")) {
			simBuilt = true;
			break;
		}
	}

	if (!simBuilt)
		return Fail("Missing _sim module, build the project first");

	fs::path script = argc > 1 ? fs::absolute(argv[1]) : root / "main.py";
	fs::current_path(exeDir);

	PyConfig config;
	PyConfig_InitPythonConfig(&config);
	config.parse_argv = 0;

	PyConfig_SetString(&config, &config.home, PYTHON_HOME);
	PyConfig_SetString(&config, &config.executable, PYTHON_EXE);
	PyConfig_SetString(&config, &config.pythonpath_env, root.wstring().c_str());
	PyConfig_SetString(&config, &config.run_filename, script.wstring().c_str());

	std::wstring scriptName = script.wstring();
	std::vector<wchar_t*> args = { scriptName.data() };
	for (int i = 2; i < argc; i++)
		args.push_back(argv[i]);

	PyConfig_SetArgv(&config, (Py_ssize_t)args.size(), args.data());

	PyStatus status = Py_InitializeFromConfig(&config);
	if (PyStatus_Exception(status)) {
		std::string error = status.err_msg ? status.err_msg : "Unknown error";
		PyConfig_Clear(&config);
		return Fail("Python failed to start: " + error);
	}
	PyConfig_Clear(&config);

	int result = Py_RunMain();
	if (result != 0)
		return Fail("Stopped with an error (see above)");

	return result;
}
