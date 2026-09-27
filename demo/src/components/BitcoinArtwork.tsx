/** Decorative identity artwork is deliberately separate from evidence charts. */
export function BitcoinArtwork({ className = '', priority = false }: { className?: string; priority?: boolean }) {
  return <div className={`bitcoin-artwork ${className}`} aria-hidden="true">
    <img src={`${import.meta.env.BASE_URL}art/bitcoin-coins.png`} alt="" width="1024" height="1024" loading={priority ? 'eager' : 'lazy'} decoding="async" draggable={false} />
  </div>;
}
