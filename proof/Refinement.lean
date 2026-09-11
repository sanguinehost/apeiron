import ApeironWord

/-!
# Refinement entry point

A compile-time smoke test that Aeneas' rooted definitions are present with the expected shape;
a drift in the generated names fails the build. The corpus-linked refinement proofs live in the
separate `proof/corpus/WordRefinement.lean` bridge package.
-/

namespace apeiron_word

#check word_mul
#check word_norm
#check word_exchange_impulse
#check word_elastic_exchange
#check word_accept_derived_impulse
#check exchange.exchange_impulse
#check exchange.elastic_exchange
#check exchange.accept_derived_impulse
#check cell.Cell.Insts.CoreCmpPartialEqCell.eq

end apeiron_word
