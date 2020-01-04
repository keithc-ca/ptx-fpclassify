# ptx-fpclassify

Eclipse OpenJ9 generates PTX code via
[PtxKernelGenerator](https://github.com/eclipse/openj9/blob/master/jcl/src/openj9.gpu/share/classes/com/ibm/gpu/PtxKernelGenerator.java),
including `testp.number.f64` which doesn't appear to work as expected.

This repository contains a small test program to explore the behavior
of all `testp.*` CUDA PTX instructions.
