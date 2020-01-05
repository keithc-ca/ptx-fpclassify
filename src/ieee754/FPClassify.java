package ieee754;

import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import com.ibm.cuda.Cuda;
import com.ibm.cuda.CudaBuffer;
import com.ibm.cuda.CudaDevice;
import com.ibm.cuda.CudaException;
import com.ibm.cuda.CudaGrid;
import com.ibm.cuda.CudaKernel;
import com.ibm.cuda.CudaModule;

public class FPClassify {

	private static void compared(CudaDevice device, CudaKernel compared, double[] values) throws CudaException {
		try (CudaBuffer buffer = new CudaBuffer(device, 20)) {
			for (int i = 0; i < values.length; ++i) {
				double lhs = values[i];
				long rawLhs = Double.doubleToRawLongBits(lhs);

				for (int j = 0; j < values.length; ++j) {
					double rhs = values[j];
					long rawRhs = Double.doubleToRawLongBits(rhs);

					buffer.copyFrom(new double[] { lhs, rhs });
					compared.launch(new CudaGrid(1, 1), buffer);

					int[] outData = new int[1];

					buffer.atOffset(Double.BYTES * 2).copyTo(outData);

					int gpuCompare = outData[0];
					int javaCompare = Double.compare(lhs, rhs) > 0 ? 1 : 0;

					System.out.format("0x%016X %-12a > 0x%016X %-12a => %d:%d%n", //
							rawLhs, lhs, rawRhs, rhs, javaCompare, gpuCompare);
				}
			}
		}
	}

	private static CudaModule load(CudaDevice device, String fileName) {
		try (FileInputStream input = new FileInputStream(fileName)) {
			return new CudaModule(device, input);
		} catch (CudaException | IOException e) {
			e.printStackTrace();
		}

		return null;
	}

	public static void main(String[] args) {
		try {
			int deviceCount = Cuda.getDeviceCount();

			if (deviceCount <= 0) {
				System.out.println("No CUDA devices available.");
				return;
			}

			CudaDevice device = new CudaDevice(0);
			CudaModule module = load(device, "sample.ptx");

			if (module == null) {
				System.out.println("Failed to load module.");
				return;
			}

			double[] values = { //
					Double.NEGATIVE_INFINITY, //
					-0.0, +0.0, //
					-1.0, +1.0, //
					Double.POSITIVE_INFINITY, //
					Double.longBitsToDouble(0x7FF0000000000001L), //
					Double.longBitsToDouble(0x7FF0000000000002L), //
					Double.longBitsToDouble(0x7FF8000000000000L), //
					Double.longBitsToDouble(0x7FFFFFFFFFFFFFFFL), //
					Double.longBitsToDouble(0xFFF0000000000001L), //
					Double.longBitsToDouble(0xFFF8000000000000L), //
					Double.longBitsToDouble(0xFFFFFFFFFFFFFFFFL) //
			};

			Arrays.sort(values);

			try {
				CudaKernel fpclassify = new CudaKernel(module, "fpclassify");

				test(device, fpclassify, values);

				System.out.println();

				CudaKernel compared = new CudaKernel(module, "compared");

				compared(device, compared, values);
			} finally {
				module.unload();
			}
		} catch (CudaException e) {
			e.printStackTrace();
		}
	}

	private static String makeClassString(long fpclass) {
		List<String> classes = new ArrayList<>();

		if ((fpclass & 0x0001) != 0) {
			classes.add("finite");
		}

		if ((fpclass & 0x0002) != 0) {
			classes.add("infinite");
		}

		if ((fpclass & 0x0100) != 0) {
			classes.add("normal");
		}

		if ((fpclass & 0x0200) != 0) {
			classes.add("subnormal");
		}

		if ((fpclass & 0x0010) != 0) {
			classes.add("number");
		}

		if ((fpclass & 0x0020) != 0) {
			classes.add("notanumber");
		}

		if ((fpclass & ~0x0333L) != 0) {
			classes.add("unknown");
		}

		return classes.stream().collect(Collectors.joining(" "));
	}

	private static void showClass(double value, long[] data) {
		long rawValue = Double.doubleToRawLongBits(value);
		long input = data[0];
		long output = data[1];
		long fpclass = data[2];
		String fpclassAsString = makeClassString(fpclass);

		if (input == rawValue && output == rawValue) {
			System.out.format("0x%016X  %-12a 0x%03X %s%n", //
					input, value, fpclass, fpclassAsString);
		} else {
			System.out.format("0x%016X -> 0x%016X  %-12a 0x%03X %s%n", //
					rawValue, output, Double.longBitsToDouble(output), fpclass, fpclassAsString);
		}
	}

	private static void test(CudaDevice device, CudaKernel fpclassify, double... values) throws CudaException {
		try (CudaBuffer buffer = new CudaBuffer(device, 24)) {
			for (double value : values) {
				double[] inData = new double[] { value, Double.NaN, Double.NaN };

				buffer.copyFrom(inData);
				fpclassify.launch(new CudaGrid(1, 1), buffer);

				long[] outData = new long[inData.length];

				buffer.copyTo(outData);

				showClass(value, outData);
			}
		}
	}

}
