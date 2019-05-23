{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE OverloadedStrings #-}


-- | Encoding MWE annotations as tree labelings.
--
-- NOTE: Currently the output of encoding is a trainig dataset element.  For
-- the sake of clarity, it would be better to first encode the annotatios as
-- tree labelings and convert them to training elements only afterwards. 


module MWE.Encode
  ( mkElem

    -- TODO: not idea we have to export those!
  , dummyRootPOS
  , dummyRootDepRel
  ) where


import           GHC.TypeNats (KnownNat)

import           Control.Monad (guard)

import qualified Data.List as List
import qualified Data.Map.Strict as M
import qualified Data.Set as S
import qualified Data.Text as T
import qualified Data.Graph as G
import qualified Data.Ord as Ord

import           Numeric.LinearAlgebra.Static.Backprop (R)

import qualified Graph
import qualified Format.Cupt as Cupt
import qualified Net.Graph as N
import           MWE.Sent


-- | Convert the given sentence to a training dataset element.  The token IDs
-- are used as vertex identifiers in the resulting graph.
--
--   * @d@ -- embedding size (dimension)
--
mkElem
  :: (KnownNat d)
  => (Cupt.MweTyp -> Bool)
    -- ^ MWE type (category) selection method
  -> Sent d
    -- ^ Input sentence
  -> N.Elem (R d)
mkElem mweSel sent0 =

  -- trace (T.unpack $ T.unwords . map Cupt.orth $ cuptSent sent0) $
  List.foldl' markMWE
    (createElem nodes arcs)
    ( filter (mweSel . Cupt.mweTyp')
    . M.elems
    . Cupt.retrieveMWEs
    -- TODO: what would happen if a MWE were marked on a token with ID range!?
    $ cuptSent sent
    )

  where

    -- Sentence with ID range tokens removed + additional dummy root token
    sent = discardMerged sent0
      { cuptSent = dummyRootTok : cuptSent sent0
      , wordEmbs = dummyRootEmb : wordEmbs sent0
      }

    -- A map from token IDs to tokens
    tokMap = M.fromList
      [ (Cupt.tokID tok, tok)
      | tok <- cuptSent sent
      ]
    -- The parent of the given token
    tokPar tok = tokMap M.! Cupt.dephead tok
    -- Is the token the root?
    isRoot tok = Cupt.dephead tok == dummyRootParID -- Cupt.TokID 0
    -- The IDs of the MWEs present in the given token
    getMweIDs tok
      = S.fromList
      . map fst
      . filter (mweSel . snd)
      $ Cupt.mwe tok

    -- Graph nodes: a list of token IDs and the corresponding vector embeddings
    nodes = do
      (tok, vec) <- zipSafe
        (cuptSent sent)
        (wordEmbs sent)
      let node = (tok, vec)
          isMwe = (not . S.null) (getMweIDs tok)
      return (tokID tok, node, isMwe)
    -- Labeled arcs of the graph
    arcs = do
      tok <- cuptSent sent
      -- Check the token is not a root
      guard . not $ isRoot tok
      let par = tokPar tok
          -- TODO: there should be no need to mark arcs as MWE in this version
          -- of `mkElem`, since `markMWEs` is executed afterwards anyway.
          --
          -- The arc value is `True` if the token and its parent are annoted as
          -- a part of the same MWE of the appropriate MWE category.
          isMwe = (not . S.null)
            (getMweIDs tok `S.intersection` getMweIDs par)
      return ((tokID tok, tokID par), isMwe)


-- | An artificial root token.
dummyRootTok :: Cupt.Token
dummyRootTok = Cupt.Token
  { tokID = Cupt.TokID 0
  , orth = ""
  , lemma = ""
  , upos = dummyRootPOS
  , xpos = ""
  , feats = M.empty
  , dephead = dummyRootParID
  , deprel = dummyRootDepRel
  , deps = ""
  , misc = ""
  , mwe = []
  }


-- | ID to refer to the parent of the artificial root node.
dummyRootParID :: Cupt.TokID
dummyRootParID = Cupt.TokID (-1)


-- | Embedding of the dummy root.
dummyRootEmb :: (KnownNat d) => R d
dummyRootEmb = 0


-- | Dummy root POS dummy label
dummyRootPOS :: T.Text
dummyRootPOS = "DUMMY-ROOT-POS"


-- | Dummy root DepRel dummy label
dummyRootDepRel :: T.Text
dummyRootDepRel = "DUMMY-ROOT-DEPREL"


-- | Remove tokens with ID ranges
discardMerged :: (KnownNat d) => Sent d -> Sent d
discardMerged sent0 
  = uncurry Sent . unzip
  $ filter cond 
  $ zip (cuptSent sent0) (wordEmbs sent0)
  where
    cond (tok, _emb) =
      case Cupt.tokID tok of
        Cupt.TokID _ -> True
        _ -> False


-- | Mark MWE in the given dataset element.
markMWE :: N.Elem d -> Cupt.Mwe -> N.Elem d
markMWE el0 mwe =
  List.foldl' markArc el0 (S.toList arcSet)
  where
    markArc el arc = el {N.arcMap = M.insert arc 1.0 (N.arcMap el)}
    arcSet =
      N.treeConnectAll
        (N.graph el0)
        (S.fromList . map tokID . S.toList $ Cupt.mweToks mwe)


-- | Create a dataset element based on nodes and labeled arcs.
--
-- Works under the assumption that incoming/outgoing arcs are ,,naturally''
-- ordered in accordance with their corresponding vertex IDs (e.g., for two
-- children nodes x, v, the node with lower ID precedes the other node).
--
-- Version of `createElem` in which node labels are explicitly handled.
--
createElem
  :: (KnownNat d)
  => [(G.Vertex, (Cupt.Token, R d), Bool)]
  -> [(Graph.Arc, Bool)]
  -> N.Elem (R d)
createElem nodes arcs0 = N.Elem
  { graph = graph
  , nodMap = _nodMap
  , arcMap = _arcMap
  , tokMap = M.fromList [(v, tok) | (v, (tok, _), _) <- nodes]
  }
  where
    -- arcs = List.sortBy (Ord.comparing _1) arcs0
    arcs = List.sortBy (Ord.comparing fst) arcs0
    vertices = [v | (v, _, _) <- nodes]
    gStr = G.buildG
      (minimum vertices, maximum vertices)
      (map fst arcs)
    graph = verify . Graph.mkAsc $ Graph.Graph
      { Graph.graphStr = gStr
      , Graph.graphInv = G.transposeG gStr
      , Graph.nodeLabelMap = M.fromList $
          map (\(x, y, _) -> (x, snd y)) nodes
      , Graph.arcLabelMap = M.fromList $
          map (\(x, _) -> (x, ())) arcs
      }
    _arcMap = M.fromList $ do
      -- (arc, _arcLab, isMwe) <- arcs
      (arc, isMwe) <- arcs
      return (arc, if isMwe then 1.0 else 0.0)
    _nodMap = M.fromList $ do
      (v, _, isMwe) <- nodes
      return (v, if isMwe then 1.0 else 0.0)
    _1 (x, _, _) = x
    verify g
      | Graph.isAsc g = g
      | otherwise = error "MWE.createElem: constructed graph not ascending!"


-- | Token ID
tokID :: Cupt.Token -> Int
tokID tok =
  case Cupt.tokID tok of
    Cupt.TokID i -> i
    Cupt.TokIDRange _ _ ->
      error "MWE.tokID: token ID ranges not supported"


----------------------------------------------
-- Utils
----------------------------------------------


-- | TODO: move to some utility module?
zipSafe :: [a] -> [b] -> [(a, b)]
zipSafe xs ys
  | length xs == length ys = zip xs ys
  | otherwise = error "zipSafe: length not the same!"
