import * as THREE from 'three'

/**
 * MeshStandardMaterial patched with Gaussian ROI glow in world space (onBeforeCompile).
 * Works with skinned meshes (BrainStem) because we inject after Three's skinning pipeline.
 */
export function createBrainGlowMaterial(): THREE.MeshStandardMaterial {
  const uniforms = {
    uCenter_ffa: { value: new THREE.Vector3() },
    uCenter_vmpfc: { value: new THREE.Vector3() },
    uCenter_ifg: { value: new THREE.Vector3() },
    uCenter_insula: { value: new THREE.Vector3() },
    uIntensity_ffa: { value: 0 },
    uIntensity_vmpfc: { value: 0 },
    uIntensity_ifg: { value: 0 },
    uIntensity_insula: { value: 0 },
    uGlowGain: { value: 1.35 },
  }

  const mat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(0x252530),
    roughness: 0.48,
    metalness: 0.12,
    envMapIntensity: 0.85,
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
          float rFfa = 11.0;
          float rVm = 13.0;
          float rIfg = 10.0;
          float rIns = 9.0;
          g += vec3(0.35, 0.75, 1.0) * neuroGlow(vNeuroGlowPos, uCenter_ffa, rFfa) * uIntensity_ffa;
          g += vec3(1.0, 0.45, 0.75) * neuroGlow(vNeuroGlowPos, uCenter_vmpfc, rVm) * uIntensity_vmpfc;
          g += vec3(1.0, 0.82, 0.35) * neuroGlow(vNeuroGlowPos, uCenter_ifg, rIfg) * uIntensity_ifg;
          g += vec3(0.45, 1.0, 0.45) * neuroGlow(vNeuroGlowPos, uCenter_insula, rIns) * uIntensity_insula;
          outgoingLight += g * uGlowGain;
        }
        #include <opaque_fragment>
        `,
      )
  }

  mat.userData.neuroUniforms = uniforms
  return mat
}
