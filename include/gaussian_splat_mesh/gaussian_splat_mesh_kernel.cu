// gaussian_splat_mesh_kernel_v2.cu
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>
#include <math.h>

#define MAX_RELATED 256
#define MAX_RELATED_GLOBAL 1024

// CUDA kernel.
// Each block processes one face (blockIdx.x is the face index).
// Threads in the block cooperatively process the pixel range [j_start, j_end) for that face.
__global__ void gaussianSplatMeshKernel(
    // Global input arrays:
    const float* __restrict__ vertices,       // (N_vertices x 3)
    const int* __restrict__ faceVerts,          // (N_faces x 3) indices into vertices
    const float* __restrict__ normals_faces,    // (N_faces x 3)
    const int* __restrict__ neighbor,           // (N_faces x N_neighbors)
    const int N_neighbors,                      // number of neighbors per face
    const int* __restrict__ faces,              // (N_splat) face id per splat
    const int* __restrict__ faces_low,          // (N_faces x 1)
    const int* __restrict__ faces_high,          // (N_faces x 1)
    const float* __restrict__ xyzs_all,         // (N_splat x 3)
    const float* __restrict__ rotations_all,    // (N_splat x 3 x 3) provided directly
    const float* __restrict__ scales_all,       // (N_splat x 2)
    const float* __restrict__ offsets_all,      // (N_splat x 1)
    const float* __restrict__ colors_all,       // (N_splat x 3)
    const float* __restrict__ opacities_all,    // (N_splat)
    // Pixel information for faces:
    const int* __restrict__ face_indices_t,     // (N_pixels) sorted face id for each pixel
    const float* __restrict__ barycentrics_t,     // (N_pixels x 3)
    const int* __restrict__ pixels_t,           // (N_pixels x 2) pixel coordinates (x,y)
    const int* __restrict__ j_starts,           // (N_faces) starting index in face_indices_t for each face
    const int* __restrict__ j_ends,             // (N_faces) ending index (non-inclusive)
    // Global outputs:
    float* __restrict__ texture_map,            // (reso x reso x 4), row-major flattening
    float* __restrict__ normal_map,             // (reso x reso x 3)
    float* __restrict__ displacement_map,       // (reso x reso x 1)
    // Other parameters:
    float* __restrict__ touch_map, 
    int reso,
    int N_splat,
    float s
)
{
    // Each block processes one face (face_i).
    int face_i = blockIdx.x;
    // Get pixel j-range for this face.
    int j_start = j_starts[face_i];
    int j_end = j_ends[face_i];

    // Shared memory for face-specific data.
    __shared__ float sh_face_center[3];
    __shared__ float sh_face_normal[3];

    // Load the face center and normal.
    if (threadIdx.x == 0) {
        int vid0 = faceVerts[face_i * 3 + 0];
        int vid1 = faceVerts[face_i * 3 + 1];
        int vid2 = faceVerts[face_i * 3 + 2];
        
        // Load vertices (each vertex has 3 floats)
        float v0[3], v1[3], v2[3];
        v0[0] = vertices[vid0 * 3 + 0]; v0[1] = vertices[vid0 * 3 + 1]; v0[2] = vertices[vid0 * 3 + 2];
        v1[0] = vertices[vid1 * 3 + 0]; v1[1] = vertices[vid1 * 3 + 1]; v1[2] = vertices[vid1 * 3 + 2];
        v2[0] = vertices[vid2 * 3 + 0]; v2[1] = vertices[vid2 * 3 + 1]; v2[2] = vertices[vid2 * 3 + 2];
        // Compute face center as average.
        sh_face_center[0] = (v0[0] + v1[0] + v2[0]) / 3.f;
        sh_face_center[1] = (v0[1] + v1[1] + v2[1]) / 3.f;
        sh_face_center[2] = (v0[2] + v1[2] + v2[2]) / 3.f;
        // Load face normal.
        sh_face_normal[0] = normals_faces[face_i * 3 + 0];
        sh_face_normal[1] = normals_faces[face_i * 3 + 1];
        sh_face_normal[2] = normals_faces[face_i * 3 + 2];

    }
    __syncthreads();

    // --- Gather candidate face ids: current face and its neighbors.
    int candidateFaces[101]; // assume at most 100 candidates.
    int nCandidates = 0;
    if (threadIdx.x == 0) {
        candidateFaces[nCandidates++] = face_i;
        for (int k = 0; k < N_neighbors; k++) {
            int nb = neighbor[face_i * N_neighbors + k];
            if (nb >= 0)
                candidateFaces[nCandidates++] = nb;
        }
    }
    __syncthreads();
    
    // --- Find related splats (whose faces match one of the candidate faces).
    // For simplicity, one thread scans through all splats.
    int related_global[MAX_RELATED_GLOBAL];
    __shared__ int nrelated_global;
    if (threadIdx.x == 0) {
        nrelated_global = 0;
        float dists[MAX_RELATED_GLOBAL];
        for (int c = 0; c < nCandidates; c++) {
            int cand = candidateFaces[c];
            int low = faces_low[cand];
            int high = faces_high[cand];
            for (int s = low; s < high; s++) {
                if (nrelated_global >= MAX_RELATED_GLOBAL) break;
                int splat_idx = faces[s];
                related_global[nrelated_global] = splat_idx;
                float gs_x = xyzs_all[splat_idx * 3 + 0];
                float gs_y = xyzs_all[splat_idx * 3 + 1];
                float gs_z = xyzs_all[splat_idx * 3 + 2];
                float dx = gs_x - sh_face_center[0];
                float dy = gs_y - sh_face_center[1];
                float dz = gs_z - sh_face_center[2];
                dists[nrelated_global++] = dx * sh_face_normal[0] + dy * sh_face_normal[1] + dz * sh_face_normal[2];
            }
        }
        // --- Bubble sort the splats by descending distance.
        for (int i = 0; i < nrelated_global - 1; i++) {
            for (int j = i + 1; j < nrelated_global; j++) {
                if (dists[i] < dists[j]) {  // swap to get descending order
                    // Swap distances.
                    float tmp = dists[i];
                    dists[i] = dists[j];
                    dists[j] = tmp;
                    // Swap corresponding splat indices.
                    int tmpIdx = related_global[i];
                    related_global[i] = related_global[j];
                    related_global[j] = tmpIdx;
                }
            }
        }

    }
    __syncthreads();
    

    // Shared arrays for related splat parameters.
    __shared__ float sh_gs_pos[MAX_RELATED][3];
    __shared__ float sh_rot[ MAX_RELATED ][9]; // each 3x3 matrix stored row-major (9 floats)
    __shared__ float sh_scales[MAX_RELATED][2];
    __shared__ float sh_offsets[MAX_RELATED];      // 1 value per splat
    __shared__ float sh_colors[MAX_RELATED][3];
    __shared__ float sh_opacities[MAX_RELATED];
    // Tangent directions computed from the rotation matrix and scales.
    __shared__ float sh_l1_tan[MAX_RELATED][3];
    __shared__ float sh_l2_tan[MAX_RELATED][3];
    __shared__ float sh_normals[MAX_RELATED][3]; // third row of rotation matrix

    for (int j = j_start + threadIdx.x; j < j_end; j += blockDim.x) {

        // Load barycentrics for this pixel.
        float bary[3];
        bary[0] = barycentrics_t[j * 3 + 0];
        bary[1] = barycentrics_t[j * 3 + 1];
        bary[2] = barycentrics_t[j * 3 + 2];

        // Compute position on the face by barycentric interpolation.
        float pos[3] = {0.f, 0.f, 0.f};
        for (int k = 0; k < 3; k++) {
            int vid = faceVerts[face_i * 3 + k];
            pos[0] += bary[k] * vertices[vid * 3 + 0];
            pos[1] += bary[k] * vertices[vid * 3 + 1];
            pos[2] += bary[k] * vertices[vid * 3 + 2];
        }
        // Splat accumulation.
        float T = 1.f;
        float outColor[3] = {0.f, 0.f, 0.f};
        float outNormal[3] = {0.f, 0.f, 0.f};
        float outDisp = 0.f;
        float median_depth = 0.f;
        uint32_t median_contributor = {-1};
        uint32_t contributor = 0;
        // --- Process related splats in chunks ---
        for (int start = 0; start < nrelated_global; start += MAX_RELATED) {
            int current_chunk = min(MAX_RELATED, nrelated_global - start);
            if (threadIdx.x == 0){
                for (int r = 0; r < current_chunk; r++) {
                    int splat_idx = related_global[start + r];
                    // Load position.
                    sh_gs_pos[r][0] = xyzs_all[splat_idx * 3 + 0];
                    sh_gs_pos[r][1] = xyzs_all[splat_idx * 3 + 1];
                    sh_gs_pos[r][2] = xyzs_all[splat_idx * 3 + 2];
                    // Load rotation matrix.
                    for (int k = 0; k < 9; k++) {
                        sh_rot[r][k] = rotations_all[splat_idx * 9 + k];
                    }
                    // Load other parameters.
                    sh_scales[r][0] = scales_all[splat_idx * 2 + 0];
                    sh_scales[r][1] = scales_all[splat_idx * 2 + 1];
                    sh_offsets[r]    = offsets_all[splat_idx];
                    sh_colors[r][0]  = colors_all[splat_idx * 3 + 0];
                    sh_colors[r][1]  = colors_all[splat_idx * 3 + 1];
                    sh_colors[r][2]  = colors_all[splat_idx * 3 + 2];
                    sh_opacities[r]  = opacities_all[splat_idx];
                
                    float l1[3] = { sh_rot[r][0] / sh_scales[r][0],
                                    sh_rot[r][3] / sh_scales[r][0],
                                    sh_rot[r][6] / sh_scales[r][0] };
                    float l2[3] = { sh_rot[r][1] / sh_scales[r][1],
                                    sh_rot[r][4] / sh_scales[r][1],
                                    sh_rot[r][7] / sh_scales[r][1] };
                    sh_l1_tan[r][0] = l1[0];
                    sh_l1_tan[r][1] = l1[1];
                    sh_l1_tan[r][2] = l1[2];
                    sh_l2_tan[r][0] = l2[0];
                    sh_l2_tan[r][1] = l2[1];
                    sh_l2_tan[r][2] = l2[2];
                    // The splat's normal (third row of rotation matrix).
                    sh_normals[r][0] = sh_rot[r][2];
                    sh_normals[r][1] = sh_rot[r][5];
                    sh_normals[r][2] = sh_rot[r][8];
                    float den = sh_normals[r][0] * sh_face_normal[0] +
                            sh_normals[r][1] * sh_face_normal[1] +
                            sh_normals[r][2] * sh_face_normal[2];
                    if (den < 0){
                    sh_normals[r][0] = -sh_normals[r][0];
                    sh_normals[r][1] = -sh_normals[r][1];
                    sh_normals[r][2] = -sh_normals[r][2];
                }
                }
            }
            __syncthreads();

            // Loop over each related splat.
            for (int r = 0; r < current_chunk; r++) {
                contributor++;
                // Compute x = ((sh_gs_pos[r] - pos) dot sh_normals[r]) / (sh_normals[r] dot sh_face_normal)
                float num = (sh_gs_pos[r][0] - pos[0]) * sh_normals[r][0] +
                            (sh_gs_pos[r][1] - pos[1]) * sh_normals[r][1] +
                            (sh_gs_pos[r][2] - pos[2]) * sh_normals[r][2];
                float den = sh_normals[r][0] * sh_face_normal[0] +
                            sh_normals[r][1] * sh_face_normal[1] +
                            sh_normals[r][2] * sh_face_normal[2];
                float x_val = num / den;

                // Compute new_diff = (pos + sh_face_normal * x_val) - sh_gs_pos[r]
                float new_diff[3] = {
                    pos[0] + sh_face_normal[0] * x_val - sh_gs_pos[r][0],
                    pos[1] + sh_face_normal[1] * x_val - sh_gs_pos[r][1],
                    pos[2] + sh_face_normal[2] * x_val - sh_gs_pos[r][2]
                };
                // if (blockIdx.x == 0 && threadIdx.x == 10){
                //     printf("num = %f, den = %f, x_val = %f\n", num, den, x_val);
                //     printf("new_diff = %f, %f, %f\n", new_diff[0], new_diff[1], new_diff[2]);
                // }
                // Compute l1_x and l2_x.
                float l1_x = new_diff[0] * sh_l1_tan[r][0] + new_diff[1] * sh_l1_tan[r][1] + new_diff[2] * sh_l1_tan[r][2];
                float l2_x = new_diff[0] * sh_l2_tan[r][0] + new_diff[1] * sh_l2_tan[r][1] + new_diff[2] * sh_l2_tan[r][2];
                float exponent = -0.5f * (l1_x * l1_x + l2_x * l2_x);
                float alpha_value = expf(exponent) * sh_opacities[r];
                // if (blockIdx.x == 0 && threadIdx.x == 10){
                //     printf("l1_x = %f, l2_x = %f, exponent = %f, alpha_value = %f\n", l1_x, l2_x, exponent, alpha_value);
                // }
                if (alpha_value < 0.01f)
                    continue;
                
                outColor[0] += alpha_value * T * sh_colors[r][0];
                outColor[1] += alpha_value * T * sh_colors[r][1];
                outColor[2] += alpha_value * T * sh_colors[r][2];
                outNormal[0] += alpha_value * T * sh_normals[r][0];
                outNormal[1] += alpha_value * T * sh_normals[r][1];
                outNormal[2] += alpha_value * T * sh_normals[r][2];
                outDisp += alpha_value * T * sh_offsets[r];
                if (T > 0.5) {
                    median_depth = sh_offsets[r];
                    median_contributor = contributor;
                }
                T *= (1.f - alpha_value);
                if (T < 0.01f)
                    break;
            }
            __syncthreads();
        }

        // Get pixel coordinates for pixel j.
        int pix_x = pixels_t[j * 2 + 0];
        int pix_y = pixels_t[j * 2 + 1];
        int pix_id = pix_y * reso + pix_x;
        texture_map[pix_id * 4 + 0] = outColor[0];
        texture_map[pix_id * 4 + 1] = outColor[1];
        texture_map[pix_id * 4 + 2] = outColor[2];
        texture_map[pix_id * 4 + 3] = 1.f - T;
        normal_map[pix_id * 3 + 0] = outNormal[0]; //  + T * sh_face_normal[0]
        normal_map[pix_id * 3 + 1] = outNormal[1]; // + T * sh_face_normal[1];
        normal_map[pix_id * 3 + 2] = outNormal[2]; //  + T * sh_face_normal[2];
        
        displacement_map[pix_id * 2 + 0] = outDisp;
        displacement_map[pix_id * 2 + 1] = median_depth;
        // Optionally, write some diagnostic value to touch_map.
        touch_map[pix_id] = (float) threadIdx.x;
    }

}
 
// C++ binding: Expose the kernel as a PyTorch extension.
std::vector<torch::Tensor> gaussian_splat_mesh_cuda(
    torch::Tensor vertices,           // float (N_vertices x 3)
    torch::Tensor faceVerts,          // int (N_faces x 3)
    torch::Tensor normals_faces,      // float (N_faces x 3)
    torch::Tensor neighbor,           // int (N_faces x N_neighbors)
    int N_neighbors,
    torch::Tensor faces,              // int (N_splat)
    torch::Tensor faces_low,          // (N_faces x 1)
    torch::Tensor faces_high,          // (N_faces x 1)
    torch::Tensor xyzs_all,           // float (N_splat x 3)
    torch::Tensor rotations_all,      // float (N_splat x 3 x 3)
    torch::Tensor scales_all,         // float (N_splat x 2)
    torch::Tensor offsets_all,        // float (N_splat x 1)
    torch::Tensor colors_all,         // float (N_splat x 3)
    torch::Tensor opacities_all,      // float (N_splat)
    torch::Tensor face_indices_t,     // int (N_pixels)
    torch::Tensor barycentrics_t,     // float (N_pixels x 3)
    torch::Tensor pixels_t,           // int (N_pixels x 2)
    torch::Tensor j_starts,           // int (N_faces)
    torch::Tensor j_ends,             // int (N_faces)
    int reso,                        // texture resolution,
    float s
)
{
    int N_faces = faceVerts.size(0);
    int N_splat = xyzs_all.size(0);
    
    auto options = vertices.options();
    auto texture_map = torch::zeros({reso, reso, 4}, options);
    auto normal_map = torch::zeros({reso, reso, 3}, options);
    auto displacement_map = torch::zeros({reso, reso, 2}, options);
    auto touch_map = torch::zeros({reso, reso, 1}, options);
    
    // Launch one block per face; choose an appropriate thread count (e.g., 256).
    dim3 blocks(N_faces);
    dim3 threads(256);
    
    gaussianSplatMeshKernel<<<blocks, threads>>>(
        vertices.contiguous().data_ptr<float>(),
        faceVerts.contiguous().data_ptr<int>(),
        normals_faces.contiguous().data_ptr<float>(),
        neighbor.contiguous().data_ptr<int>(),
        N_neighbors,
        faces.contiguous().data_ptr<int>(),
        faces_low.contiguous().data_ptr<int>(),
        faces_high.contiguous().data_ptr<int>(),
        xyzs_all.contiguous().data_ptr<float>(),
        rotations_all.contiguous().data_ptr<float>(),
        scales_all.contiguous().data_ptr<float>(),
        offsets_all.contiguous().data_ptr<float>(),
        colors_all.contiguous().data_ptr<float>(),
        opacities_all.contiguous().data_ptr<float>(),
        face_indices_t.contiguous().data_ptr<int>(),
        barycentrics_t.contiguous().data_ptr<float>(),
        pixels_t.contiguous().data_ptr<int>(),
        j_starts.contiguous().data_ptr<int>(),
        j_ends.contiguous().data_ptr<int>(),
        texture_map.contiguous().data_ptr<float>(),
        normal_map.contiguous().data_ptr<float>(),
        displacement_map.contiguous().data_ptr<float>(),
        touch_map.contiguous().data_ptr<float>(),
        reso,
        N_splat,
        s
    );
    
    return {texture_map, normal_map, displacement_map, touch_map};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gaussian_splat_mesh_cuda", &gaussian_splat_mesh_cuda, "Mesh Gaussian Splat CUDA Kernel");
}