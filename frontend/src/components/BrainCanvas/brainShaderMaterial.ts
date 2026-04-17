import * as THREE from 'three'

/**
 * Near-clear glass shell + real-time ROI glow. Tuned for high transmission (see-through) with
 * modest env/clearcoat cost vs full frosted stack.
 */
export function createBrainGlowMaterial(): THREE.MeshPhysicalMaterial {
  const uniforms = {
    uCenter_ffa: { value: new THREE.Vector3() },
    uCenter_vmpfc: { value: new THREE.Vector3() },
    uCenter_ifg: { value: new THREE.Vector3() },
    uCenter_insula: { value: new THREE.Vector3() },
    uIntensity_ffa: { value: 0 },
    uIntensity_vmpfc: { value: 0 },
    uIntensity_ifg: { value: 0 },
    uIntensity_insula: { value: 0 },
    uGlowGain: { value: 3.1 },
  }

  const mat = new THREE.MeshPhysicalMaterial({
    color: '#f4f8ff',
    roughness: 0.04,
    metalness: 0,
    transmission: 0.97,
    thickness: 0.28,
    ior: 1.52,
    attenuationColor: new THREE.Color(0xffffff),
    attenuationDistance: 4,
    envMapIntensity: 0.42,
    clearcoat: 0.12,
    clearcoatRoughness: 0.05,
    transparent: true,
    side: THREE.DoubleSide,
    depthWrite: true,
  })

  mat.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms)

    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vNeuroGlowPos;\n')
      .replace(
        '#include <project_vertex>',
        `#include <project_vertex>
        vNeuroGlowPos = ( modelMatrix * vec4( transformed, 1.0 ) ).xyz;`,
      )

    shader.fragmentShader = shader.fragmentShader
      .replace(
        '#include <common>',
        `#include <common>
        varying vec3 vNeuroGlowPos;
        uniform vec3 uCenter_ffa;
        uniform vec3 uCenter_vmpfc;
        uniform vec3 uCenter_ifg;
        uniform vec3 uCenter_insula;
        uniform float uIntensity_ffa;
        uniform float uIntensity_vmpfc;
        uniform float uIntensity_ifg;
        uniform float uIntensity_insula;
        uniform float uGlowGain;

        float neuroGlow(vec3 worldPos, vec3 center, float radius) {
          float d2 = dot(worldPos - center, worldPos - center);
          return exp(-d2 / (2.0 * radius * radius));
        }
        `,
      )
      .replace(
        '#include <opaque_fragment>',
        `
        {
          vec3 g = vec3(0.0);
          // Tight, local ROI kernels in world space so only target regions illuminate.
          float rFfa = 0.24;
          float rVm = 0.28;
          float rIfg = 0.23;
          float rIns = 0.22;
          float iFfa = pow(clamp((uIntensity_ffa - 0.12) / 0.88, 0.0, 1.0), 1.35);
          float iVm = pow(clamp((uIntensity_vmpfc - 0.12) / 0.88, 0.0, 1.0), 1.35);
          float iIfg = pow(clamp((uIntensity_ifg - 0.12) / 0.88, 0.0, 1.0), 1.35);
          float iIns = pow(clamp((uIntensity_insula - 0.12) / 0.88, 0.0, 1.0), 1.35);
          g += vec3(0.2, 0.55, 1.0) * neuroGlow(vNeuroGlowPos, uCenter_ffa, rFfa) * iFfa;
          g += vec3(0.25, 0.95, 0.45) * neuroGlow(vNeuroGlowPos, uCenter_vmpfc, rVm) * iVm;
          g += vec3(0.72, 0.28, 1.0) * neuroGlow(vNeuroGlowPos, uCenter_ifg, rIfg) * iIfg;
          g += vec3(1.0, 0.55, 0.12) * neuroGlow(vNeuroGlowPos, uCenter_insula, rIns) * iIns;
          outgoingLight += g * uGlowGain;
        }
        #include <opaque_fragment>
        `,
      )
  }

  mat.userData.neuroUniforms = uniforms
  return mat
}
