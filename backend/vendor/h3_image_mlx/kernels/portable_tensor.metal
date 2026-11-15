const int rows = x_shape[0];
const int width = x_shape[1];
const int channels = w_shape[0];
const int row = threadgroup_position_in_grid.y * BM;
const int col = threadgroup_position_in_grid.x * BN;
constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent),
                                         false, true, false);
matmul2d<desc, execution_simdgroups<4>> op;
// The caller pads the row dimension to a complete tile.
tensor<device int8_t, dextents<int, 2>, tensor_inline> a(
    (device int8_t *)x, dextents<int, 2>(width, rows));
tensor<device int8_t, dextents<int, 2>, tensor_inline> b(
    (device int8_t *)w, dextents<int, 2>(width, channels));
tensor<device int32_t, dextents<int, 2>, tensor_inline> c(
    (device int32_t *)y, dextents<int, 2>(channels, rows));
auto left = a.template slice<dynamic_extent, BM>(0, row);
auto right = b.template slice<dynamic_extent, BN>(0, col);
auto dest = c.template slice<BN, BM>(col, row);
op.run(left, right, dest);
