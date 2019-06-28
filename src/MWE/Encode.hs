{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}


-- | Encoding MWE annotations as tree labelings.


module MWE.Encode
  ( mkElem
  ) where


import           GHC.TypeNats (KnownNat)

-- import qualified Data.Graph as G
import qualified Data.Set as S
import qualified Data.Map.Strict as M
-- import qualified Data.List as List
import           Data.Monoid (Any(..))

import           Numeric.LinearAlgebra.Static.Backprop (R)

import qualified Format.Cupt as Cupt
import qualified Net.Graph as N
import qualified Graph as G
import           Graph (Labeling(..))
import           MWE.Sent (Sent(..), Node)
import qualified MWE.Sent as Sent


----------------------------------------------
-- Encoding
----------------------------------------------


-- | Convert the given sentence to a training dataset element.  The token IDs
-- are used as vertex identifiers in the resulting graph.
--
--   * @d@ -- embedding size (dimension)
--
-- TODO: The `N.Elem` type should be structured differently.  For instance, it
-- should be based on @G.Graph (Node d) ()@.
--
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent.CleanUpCfg
    -- ^ Sentence clean-up configuration
  -> Sent d
    -- ^ Input sentence
  -> N.Elem (R d)
mkElem mweSel clupCfg sent0 = N.Elem
  { graph = G.nmap Sent.emb graph
  , tokMap = fmap Sent.tok (G.nodeLabelMap graph)
  , nodMap = M.fromList $ do
      v <- G.graphNodes graph
      return (v, mark v $ nodLab lbl)
  , arcMap = M.fromList $ do
      e <- G.graphArcs graph
      return (e, mark e $ arcLab lbl)
  }
  where
    -- Remove tokens with ID ranges + add additional dummy root token
    sent = Sent.cleanUp clupCfg sent0
    -- The corresponding graph
    graph = Sent.toGraph sent
    -- And the resulting labeling
    lbl = encode mweSel sent graph
    -- Mark the given vertex/arc as MWE element or not
    mark x m = 
      case M.lookup x m of
        Just (Any True) -> 1.0
        _ -> 0.0


-- Encode the MWE annotations in the form of a graph labeling.
--
-- TODO: This function puts some silent requirements on the input sentence
-- structure:
--
-- * No tokens with ID ranges.
-- * The dependency structure is a tree
--
-- These requirements are not checked at the moment!
--
encode
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> G.Graph (Node d) ()
    -- ^ The corresponding graph representation (basically @toGraph sent@)
  -> Labeling Any
encode mweSel sent graph =
  mconcat $ map (markMWE graph) mwes
  where
    mwes
      = filter (mweSel . Cupt.mweTyp')
      . M.elems
      . Cupt.retrieveMWEs
      $ cuptSent sent


-- | Determine the labeling stemming from the given MWE.
--
-- TODO: `treeConnectAll` is originaly defined a different module than
-- `Net.Graph`. benefit from this!  `Net.Graph` should not even export it...
--
markMWE :: G.Graph (Node d) () -> Cupt.Mwe -> Labeling Any
markMWE graph mwe = Labeling
  { nodLab = mkMap nodSet
  , arcLab = mkMap arcSet
  }
  where
    nodSet = S.fromList . map Sent.tokID . S.toList $ Cupt.mweToks mwe
    arcSet = N.treeConnectAll graph nodSet
    mkMap set = M.fromList $ do
      arc <- S.toList set
      return (arc, Any True)
