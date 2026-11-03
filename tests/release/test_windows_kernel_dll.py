"""Exercise automatic function exports and runtime DLL staging on a real Windows linker."""
import os
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


class KernelCompileCommandTest(unittest.TestCase):
    def test_static_and_dll_object_compile_commands_match(self):
        # Generate MSVC commands on any host. The forced compiler is never executed.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'CMakeLists.txt').write_text('''
cmake_minimum_required(VERSION 3.28)
project(kernel_commands LANGUAGES CXX)
add_library(kernels ${KIND} kernel.cpp)
if(KIND STREQUAL "OBJECT")
  set_target_properties(kernels PROPERTIES COMPILE_PDB_NAME kernels
    COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/kernels.dir")
endif()
''')
            (root / 'kernel.cpp').write_text('int value() { return 42; }\n')
            commands = []
            for kind in ('STATIC', 'OBJECT'):
                build = root / kind
                result = subprocess.run([
                    'cmake', '-S', str(root), '-B', str(build), '-G', 'Ninja',
                    '-DCMAKE_SYSTEM_NAME=Windows', f'-DCMAKE_CXX_COMPILER={Path(sys.executable).as_posix()}',
                    '-DCMAKE_CXX_COMPILER_FORCED=ON', '-DCMAKE_CXX_COMPILER_ID_RUN=TRUE',
                    '-DCMAKE_CXX_COMPILER_ID=MSVC', '-DCMAKE_CXX_COMPILER_VERSION=19.44',
                    '-DCMAKE_CXX_COMPILER_ARCHITECTURE_ID=x64', '-DCMAKE_BUILD_TYPE=Release',
                    '-DCMAKE_EXPORT_COMPILE_COMMANDS=ON', f'-DKIND={kind}',
                ], text=True, capture_output=True, timeout=30)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                commands.append(json.loads((build / 'compile_commands.json').read_text())[0]['command'])
            self.assertEqual(commands[0], commands[1])


class KernelArchiveOrderTest(unittest.TestCase):
    def test_objects_above_the_dll_do_not_wait_for_the_exported_archive(self):
        # Object shards compile objects one by one; none may first build the kernel DLL's archive.
        module = Path(__file__).resolve().parents[2] / 'cmake/NinferTargets.cmake'
        tool = Path(sys.executable).as_posix()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'CMakeLists.txt').write_text(f'''
cmake_minimum_required(VERSION 3.28)
project(kernel_archive_order LANGUAGES CXX)
set(NINFER_WINDOWS_KERNEL_DLL ON)
include("{module.as_posix()}")
add_library(ninfer_small_t_i8 STATIC unit.cpp)
add_library(ninfer_kernels SHARED kernel.cpp)
set_target_properties(ninfer_kernels PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
ninfer_export_archive_instantiations(ninfer_kernels ninfer_small_t_i8 launch 1)
add_library(consumer STATIC consumer.cpp)
target_link_libraries(consumer PRIVATE ninfer_kernels)
''')
            for name in ('unit.cpp', 'kernel.cpp', 'consumer.cpp'):
                (root / name).write_text('int value() { return 1; }\n')
            build = root / 'build'
            result = subprocess.run([
                'cmake', '-S', str(root), '-B', str(build), '-G', 'Ninja',
                '-DCMAKE_SYSTEM_NAME=Windows', f'-DCMAKE_CXX_COMPILER={tool}',
                '-DCMAKE_CXX_COMPILER_FORCED=ON', '-DCMAKE_CXX_COMPILER_ID_RUN=TRUE',
                '-DCMAKE_CXX_COMPILER_ID=MSVC', '-DCMAKE_CXX_COMPILER_VERSION=19.44',
                '-DCMAKE_CXX_COMPILER_ARCHITECTURE_ID=x64', f'-DCMAKE_LINKER={tool}',
                f'-DCMAKE_AR={tool}', f'-DNINFER_DUMPBIN={tool}', '-DCMAKE_BUILD_TYPE=Release',
            ], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            def planned(target):
                plan = subprocess.run(['ninja', '-C', str(build), '-n', target],
                                      text=True, capture_output=True, timeout=30)
                self.assertEqual(plan.returncode, 0, plan.stdout + plan.stderr)
                return plan.stdout

            consumer = planned('CMakeFiles/consumer.dir/consumer.cpp.obj')
            self.assertIn('consumer.cpp.obj', consumer)
            self.assertNotIn('ninfer_small_t_i8', consumer)
            # The DLL still links only after the archive has written its export definitions.
            self.assertIn('ninfer_small_t_i8.lib', planned('ninfer_kernels.dll'))
            inputs = subprocess.run(['ninja', '-C', str(build), '-t', 'query', 'ninfer_kernels.dll'],
                                    text=True, capture_output=True, timeout=30).stdout
            self.assertIn('ninfer_small_t_i8_exports.def', inputs)


class ArchiveExportsTest(unittest.TestCase):
    def test_only_instantiations_are_exported_not_their_lambdas(self):
        # dumpbin lists one public symbol per line. A lambda of an instantiation carries the
        # instantiation's decorated name inside its own.
        script = Path(__file__).resolve().parents[2] / 'cmake/ArchiveExports.cmake'
        instantiation = '??$launch@$06@attention@@YAHH@Z'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            listing = root / 'listing.txt'
            listing.write_text(
                'Microsoft (R) COFF/PE Dumper Version 14.44.35207.1\n\n'
                'Dump of file units.lib\n\nFile Type: LIBRARY\n\n'
                'Archive member name at 8: /\n\n       3 public symbols\n\n'
                f'      1C2 {instantiation}\n'
                f'      1C2 ??R<lambda_1>@?1?{instantiation}@QEBA@H@Z\n'
                '      2F0 ?unmarked_helper@attention@@YAHH@Z\n')
            if os.name == 'nt':
                dumpbin = root / 'dumpbin.cmd'
                dumpbin.write_text(f'@type "{listing}"\n')
            else:
                dumpbin = root / 'dumpbin'
                dumpbin.write_text(f'#!/bin/sh\ncat "{listing}"\n')
                dumpbin.chmod(0o755)
            archive = root / 'units.lib'
            archive.write_bytes(b'!<arch>\n')
            output = root / 'exports.def'
            result = subprocess.run([
                'cmake', f'-DDUMPBIN={dumpbin}', f'-DARCHIVE={archive}', '-DSYMBOL=launch',
                '-DEXPECTED=1', f'-DOUTPUT={output}', '-P', str(script),
            ], text=True, capture_output=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(output.read_text(), f'EXPORTS\n{instantiation}\n')


def exported_names(path):
    """Names in a PE32+ image's export table."""
    data = Path(path).read_bytes()
    header = struct.unpack_from('<I', data, 0x3C)[0]
    sections = struct.unpack_from('<H', data, header + 6)[0]
    optional = struct.unpack_from('<H', data, header + 20)[0]
    directory = struct.unpack_from('<I', data, header + 24 + 112)[0]
    table = header + 24 + optional

    def offset(rva):
        for index in range(sections):
            size, address, _, raw = struct.unpack_from('<IIII', data, table + index * 40 + 8)
            if address <= rva < address + max(size, 1):
                return raw + rva - address
        raise ValueError(f'RVA {rva:#x} lies in no section')

    exports = offset(directory)
    count, names = struct.unpack_from('<I', data, exports + 24)[0], struct.unpack_from('<I', data, exports + 32)[0]
    result = []
    for index in range(count):
        start = offset(struct.unpack_from('<I', data, offset(names) + 4 * index)[0])
        result.append(data[start:data.index(b'\0', start)].decode())
    return result


@unittest.skipUnless(os.name == 'nt', 'Requires the Windows loader and MSVC')
class WindowsKernelDllTest(unittest.TestCase):
    def test_exported_bridge_runs_only_with_staged_dll(self):
        module = Path(__file__).resolve().parents[2] / 'cmake/NinferTargets.cmake'
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'CMakeLists.txt').write_text(f"""
cmake_minimum_required(VERSION 3.28)
project(kernel_dll_fixture LANGUAGES CXX)
set(NINFER_WINDOWS_KERNEL_DLL ON)
include("{module.as_posix()}")
add_library(ninfer_ggml_quants OBJECT kernel.cpp)
add_library(ninfer_device_tier OBJECT policy.cpp)
add_library(ninfer_small_t_i8 STATIC unit.cpp helper.cpp)
add_library(ninfer_kernels SHARED
  $<TARGET_OBJECTS:ninfer_ggml_quants> $<TARGET_OBJECTS:ninfer_device_tier>)
set_target_properties(ninfer_kernels PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON
  RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/kernels")
ninfer_export_archive_instantiations(ninfer_kernels ninfer_small_t_i8 launch 2)
add_executable(consumer main.cpp)
set_target_properties(consumer PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${{CMAKE_BINARY_DIR}}/apps")
target_link_libraries(consumer PRIVATE ninfer_kernels)
ninfer_stage_runtime_dlls(consumer)
""")
            (root / 'kernel.cpp').write_text(
                '#ifdef _WINDLL\n#error Object compilation must not depend on DLL linkage\n#endif\n'
                'namespace policy { int get(); }\n'
                'namespace bridge { int value() { return policy::get(); } }\n')
            (root / 'policy.cpp').write_text(
                'namespace { int cached = 0; }\n'
                'namespace policy { void set(int value) { cached = value; } int get() { return cached; } }\n')
            # No source in the archive names an export: the DLL exports its two instantiations of
            # attention::launch and nothing else of it.
            (root / 'launch.h').write_text(
                'namespace attention { int unmarked_helper(int); template <int N> int launch(int); }\n')
            (root / 'unit.cpp').write_text(
                '#include "launch.h"\n'
                'namespace attention {\n'
                'template <int N> int launch(int value) {\n'
                '    const auto add = [](int x) { return x + N; };\n'
                '    return add(unmarked_helper(value));\n'
                '}\n'
                'template int launch<7>(int);\n'
                'template int launch<8>(int);\n'
                '}\n')
            (root / 'helper.cpp').write_text(
                '#include "launch.h"\nnamespace attention { int unmarked_helper(int value) { return value * 10; } }\n')
            (root / 'main.cpp').write_text(
                'namespace bridge { int value(); } namespace policy { void set(int); }\n'
                'namespace attention { template <int N> int launch(int); }\n'
                'int main() {\n'
                '    policy::set(42);\n'
                '    return bridge::value() != 42 || attention::launch<7>(3) != 37 || attention::launch<8>(1) != 18;\n'
                '}\n')
            build = root / 'build'
            for command in (['cmake', '-S', str(root), '-B', str(build), '-G', 'Visual Studio 17 2022', '-A', 'x64'],
                            ['cmake', '--build', str(build), '--config', 'Release', '-j']):
                result = subprocess.run(command, capture_output=True, text=True, timeout=90)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            folder = build / 'apps/Release'
            exe = folder / 'consumer.exe'
            dll = folder / 'ninfer_kernels.dll'
            self.assertTrue(dll.is_file())
            names = exported_names(dll)
            self.assertEqual(sum(name.startswith('??$launch@') for name in names), 2, names)
            self.assertTrue(any('value@bridge' in name for name in names), names)
            self.assertFalse(any('unmarked_helper' in name or 'lambda' in name for name in names), names)
            env = dict(os.environ, PATH=os.environ['SystemRoot'] + r'\System32')
            self.assertEqual(subprocess.run([str(exe)], cwd=root, env=env, timeout=10).returncode, 0)
            dll.rename(dll.with_suffix('.hidden'))
            self.assertNotEqual(subprocess.run([str(exe)], cwd=root, env=env, timeout=10).returncode, 0)


if __name__ == '__main__':
    unittest.main()
