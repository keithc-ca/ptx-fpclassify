package ieee754;

import java.io.FileInputStream;
import java.io.IOException;

import com.ibm.cuda.Cuda;
import com.ibm.cuda.CudaBuffer;
import com.ibm.cuda.CudaDevice;
import com.ibm.cuda.CudaException;
import com.ibm.cuda.CudaGrid;
import com.ibm.cuda.CudaKernel;
import com.ibm.cuda.CudaModule;

public class FPClassify {

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

			try {
				CudaKernel fpclassify = new CudaKernel(module, "fpclassify");

				test(device, fpclassify, //
						Double.NEGATIVE_INFINITY, Double.POSITIVE_INFINITY, //
						-0.0, +0.0, //
						-1.0, +1.0, //
						Double.NaN, //
						Double.longBitsToDouble(0x7FF0000000000000L), //
						Double.longBitsToDouble(0x7FF0000000000001L), //
						Double.longBitsToDouble(0x7FF0000000000002L), //
						Double.longBitsToDouble(0x7FFFFFFFFFFFFFFFL), //
						Double.longBitsToDouble(0xFFF0000000000000L), //
						Double.longBitsToDouble(0xFFF0000000000001L), //
						Double.longBitsToDouble(0xFFF8000000000000L), //
						Double.longBitsToDouble(0xFFFFFFFFFFFFFFFFL) //
				);
			} finally {
				module.unload();
			}
		} catch (CudaException e) {
			e.printStackTrace();
		}
	}

	// finite     0x0001
	// infinite   0x0002
	// number     0x0010
	// notanumber 0x0020
	// normal     0x0100
	// subnormal  0x0200

	private static void showClass(long[] data) {
		long input = data[0];
		long output = data[1];
		long fpclass = data[2];

		System.out.format("0x%016X -> 0x%016X  %-12a 0x%08X%n", //
				input, output, Double.longBitsToDouble(output), fpclass);
	}

	private static void test(CudaDevice device, CudaKernel fpclassify, double... values) throws CudaException {
		try (CudaBuffer buffer = new CudaBuffer(device, 24)) {
			for (double value : values) {
				long[] data = new long[] { Double.doubleToRawLongBits(value), -2, -3 };

				buffer.copyFrom(data);
				fpclassify.launch(new CudaGrid(1, 1), buffer);
				buffer.copyTo(data);

				showClass(data);
			}
		}
	}

}
