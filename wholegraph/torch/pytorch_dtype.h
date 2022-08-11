#pragma once

#include <c10/core/ScalarType.h>

#include "data_type.h"

namespace whole_graph {

namespace pytorch {

whole_graph::WMType C10ScalarToWMType(c10::ScalarType st);
c10::ScalarType WMTypeToC10Scalar(whole_graph::WMType wmt);

}

}