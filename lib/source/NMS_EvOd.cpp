#include "NMS.hpp"
#include <riscv_vector.h>
#include <cstdint>

namespace {

constexpr int RVV_MAX_BOXES = 4096;

float bx_xmin[RVV_MAX_BOXES], bx_ymin[RVV_MAX_BOXES];
float bx_xmax[RVV_MAX_BOXES], bx_ymax[RVV_MAX_BOXES], bx_area[RVV_MAX_BOXES];


float s_xmin[RVV_MAX_BOXES], s_ymin[RVV_MAX_BOXES];
float s_xmax[RVV_MAX_BOXES], s_ymax[RVV_MAX_BOXES], s_area[RVV_MAX_BOXES];

float   s_score[RVV_MAX_BOXES];      // sort key, kept in lock-step with candidates[]
int     candidates[RVV_MAX_BOXES];
uint8_t removed[RVV_MAX_BOXES];

inline void rvv_odd_even_sort(float* key, uint32_t* idx, int n)
{
    bool prev_swapped = true;
    for (int phase = 0; phase < n; ++phase) {
        const int start  = phase & 1;
        const int npairs = (n - start) >> 1;
        bool swapped = false;

        for (int k = 0; k < npairs; ) {
            size_t vl = __riscv_vsetvl_e32m1(npairs - k);
            float*    kp = key + start + 2 * k;
            uint32_t* ip = idx + start + 2 * k;

#if __riscv_v_intrinsic >= 12000
            vfloat32m1x2_t kt = __riscv_vlseg2e32_v_f32m1x2(kp, vl);
            vuint32m1x2_t  it = __riscv_vlseg2e32_v_u32m1x2(ip, vl);
            vfloat32m1_t sL = __riscv_vget_v_f32m1x2_f32m1(kt, 0);
            vfloat32m1_t sR = __riscv_vget_v_f32m1x2_f32m1(kt, 1);
            vuint32m1_t  iL = __riscv_vget_v_u32m1x2_u32m1(it, 0);
            vuint32m1_t  iR = __riscv_vget_v_u32m1x2_u32m1(it, 1);
#else
            const ptrdiff_t stride = 2 * sizeof(float);
            vfloat32m1_t sL = __riscv_vlse32_v_f32m1(kp + 0, stride, vl);
            vfloat32m1_t sR = __riscv_vlse32_v_f32m1(kp + 1, stride, vl);
            vuint32m1_t  iL = __riscv_vlse32_v_u32m1(ip + 0, stride, vl);
            vuint32m1_t  iR = __riscv_vlse32_v_u32m1(ip + 1, stride, vl);
#endif
            // swap if R must come first:  sR > sL  ||  (sR == sL && iR < iL)
            vbool32_t gt  = __riscv_vmfgt_vv_f32m1_b32(sR, sL, vl);
            vbool32_t eq  = __riscv_vmfeq_vv_f32m1_b32(sR, sL, vl);
            vbool32_t lt  = __riscv_vmsltu_vv_u32m1_b32(iR, iL, vl);
            vbool32_t swp = __riscv_vmor_mm_b32(gt, __riscv_vmand_mm_b32(eq, lt, vl), vl);

            if (__riscv_vfirst_m_b32(swp, vl) >= 0) {          // skip the store if strip already ordered
                swapped = true;
                // vmerge(a, b, m): m ? b : a
                vfloat32m1_t nsL = __riscv_vmerge_vvm_f32m1(sL, sR, swp, vl);
                vfloat32m1_t nsR = __riscv_vmerge_vvm_f32m1(sR, sL, swp, vl);
                vuint32m1_t  niL = __riscv_vmerge_vvm_u32m1(iL, iR, swp, vl);
                vuint32m1_t  niR = __riscv_vmerge_vvm_u32m1(iR, iL, swp, vl);
#if __riscv_v_intrinsic >= 12000
                kt = __riscv_vset_v_f32m1_f32m1x2(kt, 0, nsL);
                kt = __riscv_vset_v_f32m1_f32m1x2(kt, 1, nsR);
                it = __riscv_vset_v_u32m1_u32m1x2(it, 0, niL);
                it = __riscv_vset_v_u32m1_u32m1x2(it, 1, niR);
                __riscv_vsseg2e32_v_f32m1x2(kp, kt, vl);
                __riscv_vsseg2e32_v_u32m1x2(ip, it, vl);
#else
                __riscv_vsse32_v_f32m1(kp + 0, stride, nsL, vl);
                __riscv_vsse32_v_f32m1(kp + 1, stride, nsR, vl);
                __riscv_vsse32_v_u32m1(ip + 0, stride, niL, vl);
                __riscv_vsse32_v_u32m1(ip + 1, stride, niR, vl);
#endif
            }
            k += (int)vl;
        }

        if (!swapped && !prev_swapped) break;
        prev_swapped = swapped;
    }
}

}

int vec::NonMaxSuppression(const float* boxes,
                           const float* scores,
                           int num_batches,
                           int num_classes,
                           int spatial_dim,
                           int64_t max_output_boxes_per_class,
                           float iou_threshold,
                           const float* score_threshold,
                           CenterPointBox center_point_box,
                           int64_t* selected_indices)
{
    if (spatial_dim > RVV_MAX_BOXES) return -1;
    if (max_output_boxes_per_class <= 0) return 0;

    const bool  has_thr = (score_threshold != nullptr);
    const float thr     = has_thr ? *score_threshold : 0.0f;


    const vuint8mf4_t ones = __riscv_vmv_v_x_u8mf4(1, __riscv_vsetvlmax_e8mf4());

    int num_selected = 0;

    for (int b = 0; b < num_batches; ++b) {


        const float* batch_boxes = boxes + b * spatial_dim * 4;
        for (int i = 0; i < spatial_dim; ) {
            size_t vl = __riscv_vsetvl_e32m1(spatial_dim - i);
#if __riscv_v_intrinsic >= 12000
            vfloat32m1x4_t t = __riscv_vlseg4e32_v_f32m1x4(batch_boxes + i * 4, vl);
            vfloat32m1_t f0 = __riscv_vget_v_f32m1x4_f32m1(t, 0);
            vfloat32m1_t f1 = __riscv_vget_v_f32m1x4_f32m1(t, 1);
            vfloat32m1_t f2 = __riscv_vget_v_f32m1x4_f32m1(t, 2);
            vfloat32m1_t f3 = __riscv_vget_v_f32m1x4_f32m1(t, 3);
#else
            const ptrdiff_t stride = 4 * sizeof(float);
            vfloat32m1_t f0 = __riscv_vlse32_v_f32m1(batch_boxes + i * 4 + 0, stride, vl);
            vfloat32m1_t f1 = __riscv_vlse32_v_f32m1(batch_boxes + i * 4 + 1, stride, vl);
            vfloat32m1_t f2 = __riscv_vlse32_v_f32m1(batch_boxes + i * 4 + 2, stride, vl);
            vfloat32m1_t f3 = __riscv_vlse32_v_f32m1(batch_boxes + i * 4 + 3, stride, vl);
#endif
            vfloat32m1_t xmin, ymin, xmax, ymax;
            if (center_point_box == CenterPointBox::Corners) {
                ymin = __riscv_vfmin_vv_f32m1(f0, f2, vl);
                ymax = __riscv_vfmax_vv_f32m1(f0, f2, vl);
                xmin = __riscv_vfmin_vv_f32m1(f1, f3, vl);
                xmax = __riscv_vfmax_vv_f32m1(f1, f3, vl);
            } else {
                vfloat32m1_t hw = __riscv_vfmul_vf_f32m1(f2, 0.5f, vl);
                vfloat32m1_t hh = __riscv_vfmul_vf_f32m1(f3, 0.5f, vl);
                xmin = __riscv_vfsub_vv_f32m1(f0, hw, vl);
                xmax = __riscv_vfadd_vv_f32m1(f0, hw, vl);
                ymin = __riscv_vfsub_vv_f32m1(f1, hh, vl);
                ymax = __riscv_vfadd_vv_f32m1(f1, hh, vl);
            }
            vfloat32m1_t area = __riscv_vfmul_vv_f32m1(__riscv_vfsub_vv_f32m1(xmax, xmin, vl),
                                                       __riscv_vfsub_vv_f32m1(ymax, ymin, vl), vl);
            __riscv_vse32_v_f32m1(bx_xmin + i, xmin, vl);
            __riscv_vse32_v_f32m1(bx_ymin + i, ymin, vl);
            __riscv_vse32_v_f32m1(bx_xmax + i, xmax, vl);
            __riscv_vse32_v_f32m1(bx_ymax + i, ymax, vl);
            __riscv_vse32_v_f32m1(bx_area + i, area, vl);
            i += (int)vl;
        }

        for (int c = 0; c < num_classes; ++c) {
            const float* class_scores = scores + (b * num_classes + c) * spatial_dim;

            // ==== Step 1: score filter -> candidates + their scores (sort key) ====
            int num_cand = 0;
            if (!has_thr) {                                         // everything passes
                uint32_t* out = reinterpret_cast<uint32_t*>(candidates);
                for (int i = 0; i < spatial_dim; ) {
                    size_t vl = __riscv_vsetvl_e32m1(spatial_dim - i);
                    vuint32m1_t idx = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(vl), (uint32_t)i, vl);
                    __riscv_vse32_v_u32m1(out + i, idx, vl);
                    __riscv_vse32_v_f32m1(s_score + i, __riscv_vle32_v_f32m1(class_scores + i, vl), vl);
                    i += (int)vl;
                }
                num_cand = spatial_dim;
            } else {
                for (int i = 0; i < spatial_dim; ) {
                    size_t vl = __riscv_vsetvl_e32m1(spatial_dim - i);
                    vfloat32m1_t s = __riscv_vle32_v_f32m1(class_scores + i, vl);
                    vbool32_t pass = __riscv_vmfgt_vf_f32m1_b32(s, thr, vl);
                    vuint32m1_t idx = __riscv_vadd_vx_u32m1(__riscv_vid_v_u32m1(vl), (uint32_t)i, vl);
                    vuint32m1_t  packed   = __riscv_vcompress_vm_u32m1(idx, pass, vl);
                    vfloat32m1_t packed_s = __riscv_vcompress_vm_f32m1(s, pass, vl);   // same mask -> stays paired
                    size_t cnt = __riscv_vcpop_m_b32(pass, vl);
                    __riscv_vse32_v_u32m1(reinterpret_cast<uint32_t*>(candidates + num_cand), packed, cnt);
                    __riscv_vse32_v_f32m1(s_score + num_cand, packed_s, cnt);
                    num_cand += (int)cnt;
                    i += (int)vl;
                }
            }

            if (num_cand <= 0) continue;                            // nothing passed the filter

            // ==== sort: vectorized odd-even transposition sort ====
            rvv_odd_even_sort(s_score, reinterpret_cast<uint32_t*>(candidates), num_cand);

            // ==== gather SoA into sorted order (indexed loads, byte offsets) ====
            for (int k = 0; k < num_cand; ) {
                size_t vl = __riscv_vsetvl_e32m1(num_cand - k);
                vuint32m1_t id  = __riscv_vle32_v_u32m1(reinterpret_cast<const uint32_t*>(candidates + k), vl);
                vuint32m1_t off = __riscv_vsll_vx_u32m1(id, 2, vl);   // index * 4 bytes
                __riscv_vse32_v_f32m1(s_xmin + k, __riscv_vluxei32_v_f32m1(bx_xmin, off, vl), vl);
                __riscv_vse32_v_f32m1(s_ymin + k, __riscv_vluxei32_v_f32m1(bx_ymin, off, vl), vl);
                __riscv_vse32_v_f32m1(s_xmax + k, __riscv_vluxei32_v_f32m1(bx_xmax, off, vl), vl);
                __riscv_vse32_v_f32m1(s_ymax + k, __riscv_vluxei32_v_f32m1(bx_ymax, off, vl), vl);
                __riscv_vse32_v_f32m1(s_area + k, __riscv_vluxei32_v_f32m1(bx_area, off, vl), vl);
                k += (int)vl;
            }

            // ==== Step 3: greedy selection, forward suppression ====
            for (int k = 0; k < num_cand; ) {                       // removed[0..num_cand) = 0
                size_t vl = __riscv_vsetvl_e8m1(num_cand - k);
                __riscv_vse8_v_u8m1(removed + k, __riscv_vmv_v_x_u8m1(0, vl), vl);
                k += (int)vl;
            }
            int num_kept = 0;
            for (int i = 0; i < num_cand; ++i) {
                if (removed[i]) continue;

                selected_indices[num_selected * 3 + 0] = b;
                selected_indices[num_selected * 3 + 1] = c;
                selected_indices[num_selected * 3 + 2] = candidates[i];
                num_selected++;
                if (++num_kept >= max_output_boxes_per_class) break;

                const float ay1 = s_ymin[i], ax1 = s_xmin[i];
                const float ay2 = s_ymax[i], ax2 = s_xmax[i], aa = s_area[i];

                for (int j = i + 1; j < num_cand; ) {
                    size_t vl = __riscv_vsetvl_e32m1(num_cand - j);

                    // intersection: later start, earlier end
                    vfloat32m1_t yy1 = __riscv_vfmax_vf_f32m1(__riscv_vle32_v_f32m1(s_ymin + j, vl), ay1, vl);
                    vfloat32m1_t xx1 = __riscv_vfmax_vf_f32m1(__riscv_vle32_v_f32m1(s_xmin + j, vl), ax1, vl);
                    vfloat32m1_t yy2 = __riscv_vfmin_vf_f32m1(__riscv_vle32_v_f32m1(s_ymax + j, vl), ay2, vl);
                    vfloat32m1_t xx2 = __riscv_vfmin_vf_f32m1(__riscv_vle32_v_f32m1(s_xmax + j, vl), ax2, vl);

                    vfloat32m1_t w = __riscv_vfmax_vf_f32m1(__riscv_vfsub_vv_f32m1(xx2, xx1, vl), 0.0f, vl);
                    vfloat32m1_t h = __riscv_vfmax_vf_f32m1(__riscv_vfsub_vv_f32m1(yy2, yy1, vl), 0.0f, vl);
                    vfloat32m1_t inter = __riscv_vfmul_vv_f32m1(w, h, vl);


                    vfloat32m1_t uni = __riscv_vfsub_vv_f32m1(
                        __riscv_vfadd_vf_f32m1(__riscv_vle32_v_f32m1(s_area + j, vl), aa, vl), inter, vl);
                    vfloat32m1_t iou = __riscv_vfdiv_vv_f32m1(inter, uni, vl);

                    vbool32_t bad = __riscv_vmfle_vf_f32m1_b32(uni, 0.0f, vl);
                    iou = __riscv_vfmerge_vfm_f32m1(iou, 0.0f, bad, vl);

                    vbool32_t sup = __riscv_vmfgt_vf_f32m1_b32(iou, iou_threshold, vl);
                    __riscv_vse8_v_u8mf4_m(sup, removed + j, ones, vl);   // set-only: never clears a flag

                    j += (int)vl;
                }
            }
        }
    }
    return num_selected;
}
