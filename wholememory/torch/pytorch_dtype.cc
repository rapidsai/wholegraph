#include "pytorch_dtype.h"

namespace whole_memory {

namespace pytorch {

whole_memory::WMType C10ScalarToWMType(c10::ScalarType st) {
  switch (st) {
    case c10::ScalarType::Byte: return whole_memory::WMT_Uint8;
    case c10::ScalarType::Char: return whole_memory::WMT_Int8;
    case c10::ScalarType::Short: return whole_memory::WMT_Int16;
    case c10::ScalarType::Int: return whole_memory::WMT_Int32;
    case c10::ScalarType::Long: return whole_memory::WMT_Int64;
    case c10::ScalarType::Half: return whole_memory::WMT_Half;
    case c10::ScalarType::Float: return whole_memory::WMT_Float;
    case c10::ScalarType::Double: return whole_memory::WMT_Double;
    case c10::ScalarType::BFloat16: return whole_memory::WMT_Bfloat16;
    default: std::cerr << "Scalar type " << st << " not supported.\n";
      abort();
      return whole_memory::WMT_Count;
  }
}

c10::ScalarType WMTypeToC10Scalar(whole_memory::WMType wmt) {
  switch (wmt) {
    case whole_memory::WMT_Uint8: return c10::ScalarType::Byte;
    case whole_memory::WMT_Int8: return c10::ScalarType::Char;
    case whole_memory::WMT_Int16: return c10::ScalarType::Short;
    case whole_memory::WMT_Int32: return c10::ScalarType::Int;
    case whole_memory::WMT_Int64: return c10::ScalarType::Long;
    case whole_memory::WMT_Half: return c10::ScalarType::Half;
    case whole_memory::WMT_Float: return c10::ScalarType::Float;
    case whole_memory::WMT_Double: return c10::ScalarType::Double;
    case whole_memory::WMT_Bfloat16: return c10::ScalarType::BFloat16;
    default: std::cerr << "Scalar type " << wmt << " not supported.\n";
      abort();
      return c10::ScalarType::Undefined;
  }
}

}

}