import { forwardRef } from 'react'
import './video-player.css'

type Props = {
  videoUrl: string | null
}

/**
 * Always mounts a &lt;video&gt; so the parent ref stays attached before a file is chosen.
 */
export const VideoPlayer = forwardRef<HTMLVideoElement, Props>(function VideoPlayer(
  { videoUrl },
  ref,
) {
  return (
    <div className="video-player">
      {!videoUrl ? (
        <p className="video-hint">Choose a video file below — the player stays mounted so timeline sync works as soon as you load a clip.</p>
      ) : null}
      <video
        ref={ref}
        src={videoUrl ?? undefined}
        controls
        playsInline
        className="video-el"
      />
    </div>
  )
})
