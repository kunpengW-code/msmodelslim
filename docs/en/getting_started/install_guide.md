# msModelSlim Installation Guide

## Installation Description

This document describes how to install msModelSlim. Currently, three installation methods are supported: installation from `PyPI`, installation through a `.whl` package, and building from source.

## Preparations

Prepare the Python environment: Python 3.8 or later is required.

## Building from Source

[!NOTE]Note

- When using the `msModelSlim` command line tool, do not run commands in the `msModelSlim` source code directory. Doing so may cause conflicts between the source code path and the installation path when Python imports modules, leading to command execution failures.
- If you encounter errors when installing `msmodelslim`, see the [FAQ](../appendix/faq.md) for solutions. If the issue persists, submit an [issue](https://gitcode.com/Ascend/msmodelslim/issues) with your operating environment and complete error logs attached. We will troubleshoot the issue for you as soon as possible.

### Installation on Atlas A2 Training and Inference Products and Atlas A3 Training and Inference Products

```shell
# 1. Clone the msModelSlim repository using git clone.
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2. Go to the msModelSlim directory and run the installation script.
cd msmodelslim
bash install.sh
```

### Installation on Atlas 300I Duo Products

Prerequisites: CANN has been installed and environment variables have been set.

Notes:

1. The Atlas 300I Duo card supports quantization only with a single processor on a single device.

2. To perform sparse quantization and compression, install CANN 8.2.RC1 or later.

Download the [CANN](https://www.hiascend.com/developer/download/community/result?module=cann) package. Select the AArch64 or x86_64 version based on your system architecture. For details about the installation method, see the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit).

```shell
# 1. Clone the msModelSlim repository using git clone.
git clone https://gitcode.com/Ascend/msmodelslim.git

# 2. Go to the msModelSlim directory and run the installation script.
cd msmodelslim
bash install.sh

# Note: To perform sparse quantization and compression, proceed with the following operations.
# 3. Go to the site-packages package management directory under the Python environment, where ${python_envs} specifies the Python environment path.
cd ${python_envs}/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/
# In the following example, /usr/local/ is the user directory and the Python version is 3.11.10.
cd /usr/local/lib/python3.11/site-packages/msmodelslim/pytorch/weight_compression/compress_graph/

# 4. Build the weight_compression component, where ${install_path} specifies the installation directory of the CANN software.
sudo bash build.sh ${install_path}/ascend-toolkit/latest

# 5. The build operation in the previous step generates the build directory. Grant relevant permissions to the build directory.
chmod -R 550 build
```

## Installation from `PyPI`

```bash
pip install msmodelslim
```

## Installation Through a `.whl` Package

Download the msModelSlim `.whl` software package by referring to section "Wheel Package Downloads" in [Release Notes](../appendix/release_notes.md).

After obtaining the .whl package, run the following command to install it:

```bash
sha256sum {name}.whl # Verify the .whl package. If the checksums match, the .whl package is not damaged during download.
```

```bash
pip install ./msmodelslim-{version}-py3-none-any.whl # Install the .whl package.
```

## Post-installation Configuration

For Ascend NPU devices, perform the following operations.

### Installing CANN

Download the [CANN](https://www.hiascend.com/developer/download/community/result?module=cann) package. Select the AArch64 or x86_64 version based on your system architecture. For details about the installation method, see the [CANN Installation Guide](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/softwareinst/instg/instg_0008.html?Mode=PmIns&InstallType=local&OS=Debian&Software=cannToolKit).

### Installing PTA

For details about how to install PyTorch, see configuration and installation instructions in [Ascend Extension for PyTorch](https://www.hiascend.com/document/detail/zh/Pytorch/710/configandinstg/instg/insg_0004.html).

## Uninstallation

```shell
pip uninstall msmodelslim -y
```
